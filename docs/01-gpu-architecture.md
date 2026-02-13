# GPU Architecture Fundamentals

Understanding GPU architecture is essential for writing efficient AI workloads. This chapter covers the key concepts that every AI engineer should know.

## Why GPUs for AI?

GPUs excel at AI workloads because:
1. **Massive Parallelism**: Thousands of cores vs. dozens on CPU
2. **High Memory Bandwidth**: 500+ GB/s vs. ~50 GB/s on CPU
3. **Specialized Units**: Tensor cores, matrix engines optimized for AI ops

## Execution Model

### Threads, Warps, and Blocks

```
Thread      - Single unit of execution
    ↓
Warp (32)   - Threads executing in SIMT lockstep
    ↓
Block       - Collection of warps sharing resources
    ↓
Grid        - All blocks for a kernel launch
```

**Key Insight**: All threads in a warp execute the same instruction. Divergent branches cause serialization.

### AMD vs NVIDIA Terminology

| Concept | AMD (CDNA/RDNA) | NVIDIA (Ada/Hopper) |
|---------|-----------------|---------------------|
| Execution unit | Compute Unit | Streaming Multiprocessor |
| SIMD width | Wavefront (64/32) | Warp (32) |
| Shared memory | LDS | Shared Memory |
| L1 cache | L1 | L1/Texture |
| Vector registers | VGPRs | Registers |
| Scalar registers | SGPRs | - |

## Memory Hierarchy

### Memory Types and Latency

```
                    Latency (cycles)    Bandwidth
┌─────────────────┐
│   Registers     │       1            Highest
├─────────────────┤
│   LDS/Shared    │      ~20           ~10 TB/s
├─────────────────┤
│   L1 Cache      │      ~80           High
├─────────────────┤
│   L2 Cache      │     ~200           Medium
├─────────────────┤
│   Global/HBM    │     ~400           ~500 GB/s
└─────────────────┘
```

### Memory Coalescing

When threads in a warp access consecutive memory addresses, the accesses coalesce into fewer transactions.

```cpp
// Good: Coalesced access
data[threadIdx.x]           // Each thread accesses consecutive address

// Bad: Strided access  
data[threadIdx.x * stride]  // Memory transactions multiply with stride

// Terrible: Random access
data[random_indices[threadIdx.x]]  // Each thread causes separate transaction
```

## Occupancy

**Occupancy** = Active warps / Maximum warps supported

### What Limits Occupancy?

1. **Registers per thread**: More registers → fewer concurrent warps
2. **Shared memory per block**: More shared → fewer concurrent blocks
3. **Block size**: Must divide evenly into max warps

### Finding the Sweet Spot

Higher occupancy ≠ always better. Sometimes lower occupancy with better cache utilization wins.

```
Profile to find optimal occupancy:
- rocprof --stats ./kernel
- Check "Wave Occupancy" metric
```

## Compute Units Deep Dive

### AMD CDNA Architecture (MI Series)

```
Compute Unit (CU)
├── 4 SIMD Units
│   └── 16 ALUs each (64 total)
├── Local Data Share (64KB)
├── Scalar Unit
├── L1 Cache (16KB)
└── Matrix Cores (CDNA2+)
```

### AMD RDNA Architecture (Consumer GPUs)

```
Work Group Processor (WGP)
├── 2 Compute Units
│   └── 2 SIMD32 each
├── Shared LDS (128KB)
├── Ray Accelerators
└── L1 Cache
```

## Key Performance Metrics

### Compute Utilization

```
Theoretical FLOPS = cores × clock × ops_per_cycle

For AMD MI250:
- 13,312 cores × 1.7 GHz × 2 FMA = ~45 TFLOPS (FP32)
```

### Memory Bandwidth Utilization

```
Effective Bandwidth = bytes_transferred / kernel_time
Peak Bandwidth = memory_clock × bus_width × 2 (DDR)

Utilization = Effective / Peak
```

## Optimization Checklist

- [ ] Memory accesses coalesced?
- [ ] Shared memory bank conflicts minimized?
- [ ] Occupancy reasonable for workload?
- [ ] Register pressure manageable?
- [ ] Warp divergence minimized?
- [ ] Enough work to hide latency?

## Exercises

1. **Calculate theoretical peak FLOPS** for your GPU
2. **Write a memory bandwidth test** to measure actual vs theoretical
3. **Profile a kernel** and identify if it's compute or memory bound

## Further Reading

- AMD CDNA Architecture Whitepaper
- AMD RDNA 3 Architecture Guide
- "Programming Massively Parallel Processors" by Kirk & Hwu
