# Chapter 6: Bottleneck Classification

## The Performance Bottleneck Taxonomy

Understanding why your kernel is slow is the first step to making it fast.

```
                    ┌─────────────────────┐
                    │   Performance       │
                    │   Bottleneck        │
                    └─────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
    │  Compute  │      │  Memory   │      │  Latency  │
    │  Bound    │      │  Bound    │      │  Bound    │
    └───────────┘      └───────────┘      └───────────┘
```

## 1. Compute-Bound Kernels

### Characteristics
- High VALU utilization (>60%)
- Low memory stalls
- Arithmetic intensity > roofline ridge point
- Performance scales with core count

### Identification Metrics
```
Compute Utilization = SQ_INSTS_VALU / (CU_Count × Clock × IPC_max)

If > 60%: Compute-bound
```

### Optimization Strategies
1. **Reduce instruction count**
   - Strength reduction (multiply → shift)
   - Fast math intrinsics
   - Precomputation

2. **Increase ILP**
   - Loop unrolling
   - Interleave independent operations
   - Reduce dependencies

3. **Use specialized hardware**
   - Matrix cores (WMMA/MFMA)
   - Special function units

### Example: Compute-Bound Softmax

```cpp
// Before: High instruction count
for (int i = 0; i < N; i++) {
    output[i] = exp(input[i] - max_val);
}
float sum = 0;
for (int i = 0; i < N; i++) {
    sum += output[i];
}
for (int i = 0; i < N; i++) {
    output[i] /= sum;
}

// After: Fused operations, fewer passes
float sum = 0;
for (int i = 0; i < N; i++) {
    float val = __expf(input[i] - max_val);  // Fast exp
    output[i] = val;
    sum += val;
}
float inv_sum = __frcp_rn(sum);  // Fast reciprocal
for (int i = 0; i < N; i++) {
    output[i] *= inv_sum;
}
```

## 2. Memory-Bound Kernels

### Characteristics
- High memory stalls
- Low VALU utilization
- Bandwidth close to hardware limit
- Arithmetic intensity < ridge point

### Identification Metrics
```
Memory Bound Score = Memory_Stall_Cycles / Total_Cycles

If > 40%: Memory-bound
```

### Sub-categories

#### 2a. Bandwidth-Bound
- Achieving near-peak bandwidth
- Data is too large for caches
- Streaming access patterns

```
Bandwidth Utilization = Achieved_BW / Peak_BW
If > 70%: Bandwidth-bound
```

#### 2b. Cache-Bound
- Frequent cache misses
- Data fits in cache but poor locality
- Strided or random access

```
Cache Bound = TCC_MISS_Rate > 30% with moderate BW
```

### Optimization Strategies

**For Bandwidth-Bound:**
1. Reduce data movement
   - Operator fusion
   - In-place operations
   
2. Increase arithmetic intensity
   - Batch operations
   - Data reuse with tiling

3. Compression
   - Quantization (FP16, INT8)
   - Sparse formats

**For Cache-Bound:**
1. Improve locality
   - Data layout transformation (AoS → SoA)
   - Blocked/tiled algorithms

2. Prefetching
   - Software prefetch hints
   - Larger tiles

### Example: Memory-Bound Matrix Transpose

```cpp
// Before: Non-coalesced writes (cache-bound)
__global__ void transpose_naive(float* out, float* in, int N) {
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    out[x * N + y] = in[y * N + x];  // Strided write!
}

// After: Tiled with shared memory
__global__ void transpose_tiled(float* out, float* in, int N) {
    __shared__ float tile[32][33];  // +1 padding
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    tile[threadIdx.y][threadIdx.x] = in[y * N + x];  // Coalesced read
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;  // Swapped
    y = blockIdx.x * 32 + threadIdx.y;
    
    out[y * N + x] = tile[threadIdx.x][threadIdx.y];  // Coalesced write
}
```

## 3. Latency-Bound Kernels

### Characteristics
- Low occupancy
- Instruction stalls
- Serial dependencies
- Small problem sizes

### Identification Metrics
```
Latency Bound = Occupancy < 25% AND 
                Memory_BW < 30% AND 
                Compute_Util < 30%
```

### Sub-categories

#### 3a. Occupancy-Limited
- Not enough parallelism
- Register pressure
- Shared memory limits

#### 3b. Dependency-Limited
- Long dependency chains
- Synchronization overhead
- Divergent control flow

### Optimization Strategies

**For Occupancy-Limited:**
1. Reduce register usage
   - Compiler flags (-maxrregcount)
   - Avoid register spilling

2. Launch more threads
   - Smaller work per thread
   - Multiple elements per thread

3. Reduce shared memory
   - Double buffering
   - Smaller tiles

**For Dependency-Limited:**
1. Increase ILP
   - Process multiple elements
   - Interleave operations

2. Reduce synchronization
   - Warp-synchronous code
   - Lock-free algorithms

## Bottleneck Classification Flowchart

```
Start
  │
  ▼
┌─────────────────────────────┐
│ Compute Utilization > 60%? │
└─────────────┬───────────────┘
              │
       Yes ───┼─── No
              │     │
              ▼     │
      [COMPUTE-BOUND]
              │     │
              │     ▼
              │   ┌─────────────────────────────┐
              │   │ Memory Stalls > 40%?        │
              │   └─────────────┬───────────────┘
              │                 │
              │          Yes ───┼─── No
              │                 │     │
              │                 ▼     │
              │         [MEMORY-BOUND]
              │                 │     │
              │                 │     ▼
              │                 │   ┌─────────────────┐
              │                 │   │ Occupancy < 25%?│
              │                 │   └────────┬────────┘
              │                 │            │
              │                 │     Yes ───┼─── No
              │                 │            │     │
              │                 │            ▼     │
              │                 │   [LATENCY-BOUND]
              │                 │            │     │
              │                 │            │     ▼
              │                 │            │ [MIXED/
              │                 │            │  BALANCED]
```

## Quick Reference Table

| Bottleneck | Key Metric | Target Value | Primary Fix |
|------------|-----------|--------------|-------------|
| Compute | VALU Util | < 60% | Fewer ops |
| Bandwidth | BW Util | > 70% | Fusion/quant |
| Cache | Miss Rate | > 30% | Tiling |
| Occupancy | Waves/CU | < 4 | Fewer regs |
| Dependency | IPC | < 0.5 | More ILP |

## Tools for Classification

1. **rocprof**: Hardware counters
2. **Omniperf**: Automated classification
3. **Roofline**: Visual bottleneck ID
4. **NSight**: NVIDIA equivalent
