# Chapter 10: Data Transfer Optimization

## The PCIe Bottleneck

Data transfer between CPU and GPU often dominates inference latency.

```
PCIe Bandwidth vs. GPU Compute:
┌─────────────────────────────────────────┐
│ GPU Compute: 100+ TFLOPS                │
│ GPU Memory:  1-2 TB/s                   │
│ PCIe Gen4:   ~25 GB/s (bidirectional)   │
│ PCIe Gen5:   ~50 GB/s (bidirectional)   │
└─────────────────────────────────────────┘

If your model processes 1 GB of data:
GPU processing: < 1 ms
PCIe transfer:  40 ms (!)
```

## Transfer Strategies

### 1. Pinned (Page-Locked) Memory

```cpp
// Regular allocation (pageable)
float* host_data = new float[size];
hipMemcpy(device_data, host_data, size, hipMemcpyHostToDevice);
// Slow: OS may need to copy to pinned staging buffer

// Pinned allocation
float* pinned_data;
hipHostMalloc(&pinned_data, size * sizeof(float));
hipMemcpy(device_data, pinned_data, size, hipMemcpyHostToDevice);
// Fast: Direct DMA transfer
```

Python example:
```python
import torch

# Pageable (default)
tensor = torch.randn(1000, 1000)

# Pinned memory
tensor_pinned = torch.randn(1000, 1000).pin_memory()

# Transfer to GPU
gpu_tensor = tensor_pinned.cuda(non_blocking=True)
```

### 2. Asynchronous Transfers

```cpp
// Synchronous (blocking)
hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
kernel<<<grid, block>>>(d_a, d_b);  // Waits for copy

// Asynchronous (non-blocking)
hipStream_t stream;
hipStreamCreate(&stream);

hipMemcpyAsync(d_a, h_a, size, hipMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_a, d_b);  // Overlaps!
hipMemcpyAsync(h_b, d_b, size, hipMemcpyDeviceToHost, stream);

hipStreamSynchronize(stream);  // Wait when needed
```

### 3. Double Buffering

Overlap transfer and compute:

```cpp
// Two buffers on device
float *d_buffer[2];
hipMalloc(&d_buffer[0], buffer_size);
hipMalloc(&d_buffer[1], buffer_size);

hipStream_t streams[2];
for (int i = 0; i < 2; i++) {
    hipStreamCreate(&streams[i]);
}

int current = 0;
for (int batch = 0; batch < num_batches; batch++) {
    int next = 1 - current;
    
    // Transfer next batch while processing current
    if (batch + 1 < num_batches) {
        hipMemcpyAsync(d_buffer[next], h_batches[batch + 1],
                       buffer_size, hipMemcpyHostToDevice, streams[next]);
    }
    
    // Process current batch
    kernel<<<grid, block, 0, streams[current]>>>(d_buffer[current]);
    
    // Copy back results (can overlap with next iteration)
    hipMemcpyAsync(h_results[batch], d_buffer[current],
                   result_size, hipMemcpyDeviceToHost, streams[current]);
    
    current = next;
}

hipDeviceSynchronize();
```

Timeline:
```
Stream 0: [Copy Batch 0][Kernel 0][Copy Back 0]         [Kernel 2]...
Stream 1:               [Copy Batch 1][Kernel 1][Copy Back 1]...
                        ←─ Overlap ─→
```

### 4. Zero-Copy / Unified Memory

```cpp
// Zero-copy: GPU accesses host memory directly
float* zero_copy_data;
hipHostMalloc(&zero_copy_data, size, hipHostMallocMapped);

float* d_ptr;
hipHostGetDevicePointer(&d_ptr, zero_copy_data, 0);

kernel<<<grid, block>>>(d_ptr);  // GPU reads from host memory
```

```cpp
// Unified (Managed) Memory: Automatic migration
float* unified_data;
hipMallocManaged(&unified_data, size);

// Use same pointer on CPU and GPU
for (int i = 0; i < n; i++) unified_data[i] = i;  // CPU access
kernel<<<grid, block>>>(unified_data);              // GPU access
hipDeviceSynchronize();
printf("%f\n", unified_data[0]);                    // CPU access
```

**When to use:**
- Zero-copy: Small, infrequent transfers; data accessed once
- Unified memory: Development/debugging; complex access patterns

## Optimizing Transfer Size

### Batch Multiple Small Transfers

```python
# Bad: Many small transfers
for item in items:
    item_tensor = torch.tensor(item).cuda()
    process(item_tensor)

# Good: Single large transfer
all_items = torch.tensor(items).cuda()
process(all_items)
```

### Alignment and Coalescing

```cpp
// Ensure proper alignment for efficient DMA
const size_t ALIGNMENT = 256;  // 256-byte alignment
size_t aligned_size = ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

float* aligned_buffer;
hipHostMalloc(&aligned_buffer, aligned_size);
```

## Reducing Transfer Volume

### 1. Keep Data on GPU

```python
# Bad: Transfer back and forth
for layer in model.layers:
    x = x.cuda()
    x = layer(x)
    x = x.cpu()  # Why?!
    x = preprocess(x)

# Good: Stay on GPU
x = x.cuda()
for layer in model.layers:
    x = layer(x)
x = x.cpu()  # Only at the end
```

### 2. Compression

```python
# Transfer compressed data, decompress on GPU
import zlib

# CPU side
compressed = zlib.compress(data.tobytes())
# Transfer compressed (smaller)

# GPU side (with custom kernel)
decompress_kernel<<<grid, block>>>(d_compressed, d_output);
```

### 3. Quantize Before Transfer

```python
# Transfer lower precision
# Instead of: 1000x1000 float32 = 4 MB
input_fp32 = torch.randn(1000, 1000)
input_fp16 = input_fp32.half()  # 2 MB transfer
input_gpu = input_fp16.cuda()
```

### 4. Lazy Transfer with Prefetching

```python
class PrefetchDataLoader:
    def __init__(self, data_loader, device):
        self.loader = data_loader
        self.device = device
        self.stream = torch.cuda.Stream()
        
    def __iter__(self):
        batch = None
        for next_batch in self.loader:
            if batch is not None:
                yield batch
            
            with torch.cuda.stream(self.stream):
                # Prefetch next batch
                next_batch = [
                    t.to(self.device, non_blocking=True) 
                    for t in next_batch
                ]
            
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
        
        if batch is not None:
            yield batch
```

## Multi-GPU Data Transfer

### GPU-to-GPU (Same Node)

```cpp
// Direct peer-to-peer copy (if supported)
hipDeviceCanAccessPeer(&can_access, device0, device1);

if (can_access) {
    hipDeviceEnablePeerAccess(device1, 0);
    hipMemcpyPeerAsync(d_dest, device1, d_src, device0, size, stream);
} else {
    // Fallback through host
    hipMemcpyAsync(h_temp, d_src, size, hipMemcpyDeviceToHost, stream);
    hipMemcpyAsync(d_dest, h_temp, size, hipMemcpyHostToDevice, stream);
}
```

### NVLink/Infinity Fabric

```
Direct GPU-GPU links:
- NVLink: ~600 GB/s (bidirectional)
- Infinity Fabric: ~400 GB/s
- PCIe: ~25-50 GB/s

Much faster than through CPU!
```

## Profiling Transfer Overhead

```python
import torch
import time

def profile_transfer(data_size, iterations=100):
    data = torch.randn(data_size)
    data_pinned = data.pin_memory()
    
    # Regular transfer
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = data.cuda()
        torch.cuda.synchronize()
    regular_time = (time.time() - start) * 1000 / iterations
    
    # Pinned transfer
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = data_pinned.cuda()
        torch.cuda.synchronize()
    pinned_time = (time.time() - start) * 1000 / iterations
    
    size_mb = data.numel() * 4 / 1024 / 1024
    print(f"Data size: {size_mb:.2f} MB")
    print(f"Regular: {regular_time:.3f} ms ({size_mb/regular_time*1000:.2f} GB/s)")
    print(f"Pinned:  {pinned_time:.3f} ms ({size_mb/pinned_time*1000:.2f} GB/s)")
```

## Best Practices Summary

1. **Use pinned memory**: 2-3x faster transfers
2. **Overlap compute and transfer**: Double buffering
3. **Minimize transfer count**: Batch operations
4. **Keep data on GPU**: Avoid round-trips
5. **Use appropriate precision**: FP16/INT8 when possible
6. **Profile transfers**: Often hidden bottleneck
7. **Consider PCIe topology**: NUMA placement matters
