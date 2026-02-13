# Chapter 8: Operator Fusion

## What is Operator Fusion?

Operator fusion combines multiple operations into a single kernel, reducing memory traffic and kernel launch overhead.

```
Before Fusion:                    After Fusion:
┌────────┐                        ┌────────────────────┐
│ MatMul │ → Memory               │                    │
└────────┘     ↓                  │  MatMul + BiasAdd  │
┌────────┐     ↑                  │     + ReLU         │
│BiasAdd │ → Memory               │   (Single Kernel)  │
└────────┘     ↓                  │                    │
┌────────┐     ↑                  └────────────────────┘
│  ReLU  │ → Memory               
└────────┘                        Memory accesses: 3x → 1x
```

## Why Fusion Matters

### Memory Bandwidth Bottleneck

Modern AI workloads are often memory-bound:
- GPU memory bandwidth: 1-2 TB/s
- Compute throughput: 100+ TFLOPS
- Arithmetic intensity needed: 50+ FLOPs/byte

### Kernel Launch Overhead

Each kernel launch costs:
- ~5-20 μs on modern GPUs
- Adds up with hundreds of operators
- Prevents continuous compute utilization

## Common Fusion Patterns

### 1. Conv-BN-ReLU Fusion

```
Original: Conv2D → BatchNorm → ReLU

Fused formula:
y = ReLU(γ * (x * w + b - μ) / σ + β)
y = ReLU(x * w' + b')

where:
w' = γ * w / σ
b' = γ * (b - μ) / σ + β
```

### 2. MatMul-Add-Activation Fusion

```cpp
// Before: 3 kernels
C = A @ B           // Kernel 1
C = C + bias        // Kernel 2  
C = relu(C)         // Kernel 3

// After: 1 kernel
__global__ void fused_gemm_bias_relu(
    float* A, float* B, float* bias, float* C,
    int M, int N, int K
) {
    // Compute C = A @ B + bias in shared memory
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    sum += bias[col];
    
    // Apply ReLU inline
    C[row * N + col] = fmaxf(0.0f, sum);
}
```

### 3. Attention Fusion (Flash Attention)

```
Standard Attention:
Q, K, V = Linear(x)     // Kernel 1-3
S = Q @ K.T              // Kernel 4
S = S / sqrt(d)          // Kernel 5
P = softmax(S)           // Kernel 6-7
O = P @ V                // Kernel 8

Flash Attention: All in 1 kernel!
- Never materializes S or P matrices
- O(N) memory instead of O(N²)
- 2-4x faster in practice
```

### 4. LayerNorm Fusion

```cpp
// Fused: mean, variance, normalize, scale, shift
__global__ void fused_layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int hidden_size,
    float eps
) {
    __shared__ float s_mean, s_var;
    
    // Compute mean (parallel reduction)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += input[blockIdx.x * hidden_size + i];
    }
    // Reduce sum...
    
    // Compute variance (parallel reduction)
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = input[blockIdx.x * hidden_size + i] - s_mean;
        var_sum += diff * diff;
    }
    // Reduce variance...
    
    // Normalize, scale, shift
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = input[blockIdx.x * hidden_size + i];
        output[blockIdx.x * hidden_size + i] = 
            gamma[i] * (x - s_mean) / sqrtf(s_var + eps) + beta[i];
    }
}
```

## Fusion in Deep Learning Frameworks

### ONNX Runtime Graph Optimizations

```python
import onnxruntime as ort

# Enable all optimizations (includes fusion)
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Specific transformations include:
# - ConvBnFusion
# - MatMulBnFusion
# - ReluFusion
# - GemmActivationFusion
# - LayerNormFusion
# - AttentionFusion
```

### TensorRT Fusion

```cpp
// TensorRT automatically fuses operations
// Enable during builder configuration
config->setFlag(BuilderFlag::kFP16);  // Also enables FP16 fusion
config->setFlag(BuilderFlag::kSTRICT_TYPES);

// Common TensorRT fusions:
// - Convolution + Bias + Activation
// - Fully Connected + Bias + Activation
// - Scale + Offset + Power
```

### PyTorch 2.0 Compile

```python
import torch

# torch.compile applies fusion automatically
model = torch.compile(model, mode="reduce-overhead")

# Inspecting generated kernels
torch._dynamo.config.verbose = True
torch._inductor.config.debug = True
```

## Manual Fusion Strategies

### 1. Identifying Fusion Opportunities

```python
# Analyze ONNX graph for fusion patterns
import onnx
from onnx import helper

def find_fusable_patterns(model):
    patterns = []
    graph = model.graph
    
    for i, node in enumerate(graph.node):
        if node.op_type == "Conv":
            next_node = find_consumer(graph, node.output[0])
            if next_node and next_node.op_type == "BatchNormalization":
                next_next = find_consumer(graph, next_node.output[0])
                if next_next and next_next.op_type == "Relu":
                    patterns.append(("Conv-BN-ReLU", i))
    
    return patterns
```

### 2. Writing Custom Fused Kernels

```cpp
// Example: Fused GELU activation
// GELU(x) = x * Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

__global__ void fused_bias_gelu(
    const float* input,
    const float* bias,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx] + bias[idx % bias_size];
    
    // Fast GELU approximation
    const float c = 0.044715f;
    const float sqrt_2_pi = 0.7978845608f;
    
    float x3 = x * x * x;
    float inner = sqrt_2_pi * (x + c * x3);
    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}
```

## Fusion Limitations

### Register Pressure

```
More operations per kernel = more registers needed
Limited registers → reduced occupancy → worse performance

Solution: Balance fusion depth vs. occupancy
```

### Memory Access Patterns

```
Can only fuse operations with compatible access patterns:
✓ Element-wise after MatMul: Same output layout
✗ Transpose after MatMul: Incompatible patterns
```

### Dependency Chains

```
Cannot fuse operations with external dependencies:
X → A → Y
    ↓
    B → Z    // A cannot be fused if B needs intermediate result
```

## Measuring Fusion Impact

```python
import time

# Without fusion
start = time.time()
for _ in range(100):
    x = model_unfused(input)
unfused_time = time.time() - start

# With fusion
start = time.time()
for _ in range(100):
    x = model_fused(input)
fused_time = time.time() - start

print(f"Speedup: {unfused_time / fused_time:.2f}x")
print(f"Time saved: {(unfused_time - fused_time) * 10:.2f} ms per inference")
```

## Best Practices

1. **Use framework optimizations**: Enable max optimization levels
2. **Profile first**: Identify memory-bound operators
3. **Custom kernels sparingly**: Only when frameworks can't fuse
4. **Test accuracy**: Fused math may differ slightly
5. **Consider occupancy**: Don't over-fuse
