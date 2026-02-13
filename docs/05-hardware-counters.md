# Chapter 5: Hardware Performance Counters

## Understanding GPU Hardware Counters

Hardware performance counters are the most precise way to understand what's happening inside your GPU. They provide direct measurements from the hardware itself.

## AMD GPU Counter Categories

### 1. Compute Unit (CU) Counters

```
SQ_WAVES          - Total wavefronts launched
SQ_INSTS_VALU     - Vector ALU instructions executed
SQ_INSTS_SALU     - Scalar ALU instructions executed
SQ_INSTS_LDS      - Local Data Share instructions
SQ_INSTS_SMEM     - Scalar memory instructions
SQ_WAIT_INST_LDS  - Cycles waiting for LDS
SQ_ACTIVE_INST_VALU - Cycles with active VALU
```

### 2. Memory Counters

```
TCP_READ_TAGCONFLICT_STALL_CYCLES  - Cache tag conflicts
TCP_PENDING_STALL_CYCLES           - Pending request stalls
TCC_HIT                            - L2 cache hits
TCC_MISS                           - L2 cache misses
TCC_EA_RDREQ                       - Read requests to memory
TCC_EA_WRREQ                       - Write requests to memory
```

### 3. Instruction Mix Counters

```
SQ_INSTS_VMEM_WR  - Vector memory writes
SQ_INSTS_VMEM_RD  - Vector memory reads
SQ_INSTS_FLAT     - Flat address space instructions
SQ_INSTS_BRANCH   - Branch instructions
```

## Collecting Hardware Counters

### Using rocprof

```bash
# Basic counter collection
rocprof --stats -i counters.txt ./my_kernel

# Counter input file (counters.txt)
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS
pmc: TCP_READ_TAGCONFLICT_STALL_CYCLES TCC_HIT TCC_MISS
```

### Using Omniperf

```bash
# Full profiling pass
omniperf profile -n my_profile -- ./my_kernel

# Analyze results
omniperf analyze -p my_profile/
```

## Key Metrics to Monitor

### 1. Occupancy

```
Occupancy = Active Wavefronts / Maximum Wavefronts per CU

Target: >50% for compute-bound kernels
        Higher for memory-bound kernels
```

### 2. IPC (Instructions Per Cycle)

```
IPC = Instructions Executed / Cycles

VALU IPC: Target ~1.0 (theoretical max varies by arch)
High IPC = Good instruction-level parallelism
```

### 3. Memory Bandwidth Utilization

```
Bandwidth Utilization = Achieved Bandwidth / Peak Bandwidth

HBM2e Peak (MI210): ~1.6 TB/s
GDDR6 Peak:         ~500-600 GB/s
```

### 4. Cache Hit Rates

```
L1 Hit Rate = TCP_HIT / (TCP_HIT + TCP_MISS)
L2 Hit Rate = TCC_HIT / (TCC_HIT + TCC_MISS)

Target: >80% for data-reuse kernels
```

## Counter Analysis Example

### Vector Addition Kernel

```cpp
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
```

Expected Counter Profile:
```
SQ_WAVES:        High (one per 64 threads)
SQ_INSTS_VALU:   Low (only one FADD per element)
SQ_INSTS_VMEM:   High (3 memory ops: 2 reads, 1 write)
TCC_HIT:         Low (streaming access, no reuse)

Diagnosis: Memory-bound kernel
Optimization: Use vectorized loads (float4)
```

### Matrix Multiply Kernel

```cpp
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

Expected Counter Profile:
```
SQ_WAVES:        Medium
SQ_INSTS_VALU:   Very High (N FMAs per output)
SQ_INSTS_VMEM:   Very High (2N reads per output)
TCC_MISS:        Very High (poor cache utilization)

Diagnosis: Memory-bound despite compute intensity
Optimization: Tiled implementation with shared memory
```

## Performance Counter Formulas

### Arithmetic Intensity

```
AI = FLOPs / Bytes Transferred
   = SQ_INSTS_VALU / (TCP_READ_DATA + TCP_WRITE_DATA) × sizeof(float)
```

### Memory Stall Ratio

```
Stall Ratio = Memory Wait Cycles / Total Cycles
            = (TCP_PENDING_STALL_CYCLES) / (GPU_BUSY_CYCLES)
```

### Wavefront Efficiency

```
Efficiency = Useful Threads / Total Threads
           = (Valid work-items) / (Wavefront Size × Wavefronts)
```

## Hardware Counter Limitations

1. **Multiplexing**: Can't collect all counters simultaneously
2. **Overhead**: Some counters add measurement overhead
3. **Sampling**: May not capture all kernel launches
4. **Architecture-specific**: Counter names vary across GPUs

## Best Practices

1. **Baseline first**: Establish performance baseline
2. **Compare iterations**: Track changes across optimizations
3. **Focus on bottlenecks**: Don't optimize non-critical paths
4. **Validate with multiple counters**: Cross-reference data
5. **Consider roofline**: Put counters in context
