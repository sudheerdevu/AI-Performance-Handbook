# Chapter 7: Quantization for AI Inference

## What is Quantization?

Quantization reduces the numerical precision of model weights and activations from floating-point (FP32) to lower-bit representations (FP16, INT8, INT4).

```
FP32 (32 bits) → FP16 (16 bits) → INT8 (8 bits) → INT4 (4 bits)
     ↓               ↓               ↓               ↓
  Baseline        2x speedup      4x speedup      8x speedup*
                  2x memory       4x memory       8x memory
```

*Theoretical maximum; actual gains depend on hardware support

## Quantization Types

### 1. Post-Training Quantization (PTQ)

Apply quantization after training without retraining.

```python
# Example: ONNX Runtime dynamic quantization
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input="model_fp32.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QUInt8
)
```

**Pros:**
- Fast and easy
- No training data needed
- Works for most models

**Cons:**
- May lose accuracy
- Limited optimization

### 2. Quantization-Aware Training (QAT)

Simulate quantization during training to recover accuracy.

```python
# PyTorch QAT example
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# Fine-tune with quantization simulation
for epoch in range(num_epochs):
    train(model, train_loader)
    
# Convert to quantized model
quantized_model = torch.quantization.convert(model.eval())
```

**Pros:**
- Higher accuracy recovery
- Better optimization potential

**Cons:**
- Requires training infrastructure
- More time-consuming

## Quantization Granularity

### Per-Tensor Quantization

Single scale factor for entire tensor:
```
Q(x) = round(x / scale) + zero_point
```

### Per-Channel Quantization

Different scale per output channel:
```
Q(x[c]) = round(x[c] / scale[c]) + zero_point[c]
```

### Per-Group Quantization

Scale per group of weights (common in LLMs):
```
Group size: 32, 64, or 128 elements per scale factor
Better accuracy for weight-only quantization
```

## FP16 Inference

### Mixed Precision

```cpp
// ROCm: Enable FP16 compute
rocblas_set_math_mode(handle, rocblas_xf32_xdl_math_op);

// ONNX Runtime
session_options.add_session_config_entry(
    "session.use_fp16", "1"
);
```

### Hardware Support

| GPU | FP16 Support | FP16 Tensor Cores |
|-----|--------------|-------------------|
| AMD MI210 | Native | Matrix Cores |
| AMD RX 7900 | Native | WMMA |
| NVIDIA A100 | Native | Tensor Cores |

## INT8 Inference

### Calibration for PTQ

```python
from onnxruntime.quantization import CalibrationDataReader, quantize_static

class MyCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_data):
        self.data = iter(calibration_data)
    
    def get_next(self):
        return next(self.data, None)

quantize_static(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    calibration_data_reader=MyCalibrationReader(data),
    quant_format=QuantFormat.QDQ
)
```

### Symmetric vs Asymmetric

**Symmetric**: zero_point = 0
```
scale = max(abs(min), abs(max)) / 127
Q(x) = round(x / scale)
Range: [-127, 127]
```

**Asymmetric**: zero_point ≠ 0
```
scale = (max - min) / 255
zero_point = round(-min / scale)
Q(x) = round(x / scale) + zero_point
Range: [0, 255]
```

## LLM-Specific Quantization

### Weight-Only Quantization

For large language models, quantize only weights:

```
W4A16: 4-bit weights, 16-bit activations
GPTQ, AWQ, GGML formats
```

### GPTQ Algorithm

```python
# Simplified GPTQ concept
# Quantize weights layer by layer
# Use Hessian information to minimize error

for layer in model.layers:
    H = compute_hessian(layer, calibration_data)
    Q, scale = gptq_quantize(layer.weight, H, bits=4)
    layer.weight = dequantize(Q, scale)  # Keep in higher precision
```

### Block-wise Quantization

```
Common in LLMs:
- Block size: 32, 64, 128
- Each block has its own scale
- Better accuracy for outliers
```

## Accuracy Recovery Techniques

### 1. Outlier Handling

```python
# SmoothQuant: migrate outliers from activations to weights
# Before: Y = X @ W  (X has outliers)
# After:  Y = (X @ diag(s)^-1) @ (diag(s) @ W)

scale = activation_outlier_scale / weight_tolerance
X_smooth = X / scale
W_smooth = W * scale
```

### 2. Mixed-Precision Quantization

Keep sensitive layers at higher precision:
```python
sensitive_layers = identify_sensitive_layers(model, val_data)
for layer in model:
    if layer in sensitive_layers:
        quantize(layer, bits=16)  # Higher precision
    else:
        quantize(layer, bits=8)   # Lower precision
```

### 3. Fine-tuning After Quantization

```python
# Knowledge distillation from FP32 teacher
loss = alpha * task_loss + beta * KL_div(teacher_logits, student_logits)
```

## Performance Impact

### Memory Bandwidth Savings

```
Model Size Reduction:
FP32 → FP16: 50% reduction
FP32 → INT8: 75% reduction
FP32 → INT4: 87.5% reduction

Example: LLaMA-7B
FP32: 28 GB
FP16: 14 GB
INT8: 7 GB
INT4: 3.5 GB
```

### Compute Throughput

```
Theoretical speedup (bandwidth-bound):
FP16: 2x base
INT8: 4x base (with INT8 tensor cores)
INT4: 8x base (limited hardware support)

Practical numbers vary by kernel and hardware
```

## Best Practices

1. **Start with FP16**: Easy win, minimal accuracy loss
2. **Use calibration data**: Representative of production traffic
3. **Validate accuracy**: Test on held-out data
4. **Profile thoroughly**: Ensure actual speedup
5. **Consider QAT**: For sensitive models

## Tools and Libraries

| Tool | Vendor | Formats |
|------|--------|---------|
| ONNX RT Quantization | Microsoft | INT8, QDQ |
| TensorRT | NVIDIA | INT8, FP16 |
| MIGraphX | AMD | INT8, FP16 |
| llama.cpp | Community | GGML, GGUF |
| AutoGPTQ | Community | GPTQ |
| AutoAWQ | Community | AWQ |
