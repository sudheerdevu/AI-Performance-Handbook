# AI Performance Cheat Sheet

Quick reference for common performance optimization scenarios.

## Profiling Commands

### ROCm (AMD)

```bash
# Basic kernel stats
rocprof --stats ./app

# Hardware counters
rocprof -i counters.txt ./app

# Trace timeline
rocprof --hip-trace --hsa-trace ./app

# Memory analysis
rocprof -i memory_counters.txt ./app

# Omniperf (comprehensive)
omniperf profile -n my_analysis -- ./app
omniperf analyze -p workloads/my_analysis/
```

### CUDA (NVIDIA)

```bash
# Nsight Systems timeline
nsys profile -o report ./app

# Nsight Compute kernel analysis
ncu --set full ./app

# Memory throughput
ncu --metrics l2_utilization ./app
```

## Bottleneck Classification

### Memory Bound Indicators
- High memory throughput, low compute utilization
- L2 cache hit rate < 50%
- Memory pipeline stalls

**Fix**: Improve locality, reduce data movement, use compression

### Compute Bound Indicators
- High ALU utilization
- Low memory throughput
- FP32/FP16/INT8 unit saturation

**Fix**: Reduce precision, instruction mix optimization

### Latency Bound Indicators
- Low both compute and memory utilization  
- High kernel launch overhead
- Synchronization stalls

**Fix**: Increase parallelism, batch operations, async execution

## Memory Optimization

### Coalescing Rules

```cpp
// GOOD: Sequential access
arr[threadIdx.x]

// BAD: Strided access (32x more transactions)
arr[threadIdx.x * stride]

// SOLUTION: Transpose or pad data
```

### LDS/Shared Memory Tips

```cpp
// Bank conflict free access
__shared__ float data[SIZE + 1];  // +1 padding

// Broadcast is free
float val = shared_data[0];  // All threads read same address = 1 transaction
```

## Kernel Optimization Patterns

### Loop Unrolling

```cpp
// Manual unroll for predictable loops
#pragma unroll 4
for (int i = 0; i < 16; i += 4) {
    sum += data[i] + data[i+1] + data[i+2] + data[i+3];
}
```

### Register Blocking

```cpp
// Load to registers for reuse
float a0 = A[...], a1 = A[...], a2 = A[...], a3 = A[...];
// Use a0, a1, a2, a3 multiple times before loading more
```

### Warp-Level Operations

```cpp
// Efficient intra-warp communication
float sum = warp_reduce(val);  // Uses __shfl_down

// No need for __syncthreads() within warp
```

## Quick Reference Tables

### Memory Bandwidth by GPU

| GPU | Memory | Bandwidth |
|-----|--------|-----------|
| RTX 4090 | GDDR6X | 1 TB/s |
| MI250X | HBM2e | 3.2 TB/s |
| H100 | HBM3 | 3.35 TB/s |
| MI300X | HBM3 | 5.3 TB/s |

### Typical Operation Costs

| Operation | Cycles | Notes |
|-----------|--------|-------|
| FP32 ADD/MUL | 4 | Single precision |
| FP32 FMA | 4 | + and × together |
| FP16 FMA | 2 | Use when possible |
| INT8 MAC | 1 | For inference |
| Shared load | 20 | No conflict |
| Global load | 400 | Cached |

### Precision Impact

| Precision | Throughput | Memory | Accuracy |
|-----------|------------|--------|----------|
| FP32 | 1x | 1x | Baseline |
| FP16 | 2-4x | 0.5x | ~0.1% loss |
| INT8 | 4-8x | 0.25x | ~1% loss |
| INT4 | 8-16x | 0.125x | Model-dependent |

## Common Pitfalls

### ❌ Don't Do This

```cpp
// Atomic in hot loop
atomicAdd(&global_sum, my_val);  // Serialize all threads

// Branch divergence in inner loop
if (threadIdx.x % 2 == 0) { ... }  // 50% warp efficiency

// Uncoalesced global access
data[threadIdx.y * WIDTH + threadIdx.x * STRIDE]
```

### ✅ Do This Instead

```cpp
// Reduce locally first, single atomic at end
__shared__ float local_sum;
// ... parallel reduction to local_sum
if (threadIdx.x == 0) atomicAdd(&global_sum, local_sum);

// Process independently in each branch
// Or reorganize data to avoid divergence

// Transpose for coalesced access
data[threadIdx.y * WIDTH + threadIdx.x]
```

## Performance Estimation

### Roofline Quick Math

```
Peak FLOPS: cores × clock × 2 (FMA)
Peak BW: memory_clock × bus_width × 2 (DDR)
Ridge Point: FLOPS / BW = arithmetic_intensity threshold

If AI < ridge point: memory bound
If AI > ridge point: compute bound
```

### Latency Estimation

```
Kernel latency ≈ max(compute_time, memory_time)

compute_time = ops / (flops × utilization)
memory_time = bytes / (bandwidth × utilization)
```

## Debugging Performance Issues

```
Issue: Kernel is slow but GPU utilization is low

Check:
1. Kernel launch overhead (too many small kernels?)
2. Host-device sync points (unnecessary synchronization?)
3. Memory copy stalls (overlap compute with transfer?)
4. Insufficient parallelism (increase work per launch)

Issue: High GPU utilization but still slow

Check:
1. Wrong bottleneck assumption (profile again)
2. Algorithm complexity (can you reduce O(n²) to O(n)?)
3. Data layout mismatch (NCHW vs NHWC?)
4. Precision overkill (do you need FP32?)
```

## Links

- [ROCm Documentation](https://rocm.docs.amd.com)
- [rocprof User Guide](https://rocm.docs.amd.com/projects/rocprofiler)
- [Omniperf Documentation](https://rocm.docs.amd.com/projects/omniperf)
