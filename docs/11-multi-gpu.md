# Chapter 11: Multi-GPU Inference

## When to Use Multi-GPU

```
Single GPU Limit:
- Model doesn't fit in VRAM
- Latency too high for single GPU
- Throughput requirements exceed capacity

Multi-GPU Solution:
- Model parallelism: Split model across GPUs
- Data parallelism: Replicate model, split data
- Pipeline parallelism: Split model into stages
```

## Model Parallelism

### Tensor Parallelism

Split individual operations across GPUs:

```
Example: Large Matrix Multiply
A × B = C

Tensor Parallel on 2 GPUs:
GPU 0: A × B[:, :N/2] = C[:, :N/2]
GPU 1: A × B[:, N/2:] = C[:, N/2:]

Result: Concatenate or AllGather
```

```python
# PyTorch tensor parallelism
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

# Parallelize linear layers
parallelize_plan = {
    "attention.q_proj": ColwiseParallel(),
    "attention.k_proj": ColwiseParallel(),
    "attention.v_proj": ColwiseParallel(),
    "attention.out_proj": RowwiseParallel(),
    "mlp.fc1": ColwiseParallel(),
    "mlp.fc2": RowwiseParallel(),
}

model = parallelize_module(model, device_mesh, parallelize_plan)
```

### Layer/Pipeline Parallelism

Split layers across GPUs:

```
12-layer model on 4 GPUs:
GPU 0: Layers 0-2    [Embedding]
GPU 1: Layers 3-5    
GPU 2: Layers 6-8    
GPU 3: Layers 9-11   [Output Head]

Forward: Data flows GPU 0 → 1 → 2 → 3
Backward: Gradients flow GPU 3 → 2 → 1 → 0
```

```python
# Simple layer sharding
import torch.nn as nn

class PipelinedModel(nn.Module):
    def __init__(self, layers, devices):
        super().__init__()
        self.layers = nn.ModuleList()
        self.devices = devices
        
        layers_per_device = len(layers) // len(devices)
        for i, device in enumerate(devices):
            start = i * layers_per_device
            end = start + layers_per_device
            stage = nn.Sequential(*layers[start:end]).to(device)
            self.layers.append(stage)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = x.to(self.devices[i])
            x = layer(x)
        return x
```

## Data Parallelism for Inference

### Model Replication

```python
import torch.nn.parallel as parallel

# Replicate model across GPUs
model = MyModel().cuda()
model = parallel.DataParallel(model, device_ids=[0, 1, 2, 3])

# Batch is automatically split
batch = load_batch()  # [128, ...]
outputs = model(batch)  # Each GPU gets 32 samples
```

### Load Balancing

```python
class InferenceLoadBalancer:
    def __init__(self, model_class, num_gpus):
        self.models = []
        self.queues = []
        
        for i in range(num_gpus):
            model = model_class().cuda(i)
            model.eval()
            self.models.append(model)
            self.queues.append(asyncio.Queue())
        
        # Start worker for each GPU
        for i in range(num_gpus):
            asyncio.create_task(self._worker(i))
    
    async def _worker(self, gpu_id):
        model = self.models[gpu_id]
        queue = self.queues[gpu_id]
        
        while True:
            request, future = await queue.get()
            with torch.cuda.device(gpu_id):
                result = model(request)
            future.set_result(result)
    
    async def infer(self, request):
        # Route to least busy GPU
        gpu_id = self._select_gpu()
        future = asyncio.Future()
        await self.queues[gpu_id].put((request, future))
        return await future
    
    def _select_gpu(self):
        # Simple: round-robin or shortest queue
        return min(range(len(self.queues)), 
                   key=lambda i: self.queues[i].qsize())
```

## Communication Patterns

### AllReduce

```python
# Sum values across all GPUs
import torch.distributed as dist

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Each GPU now has the sum of all GPUs' tensors
```

### AllGather

```python
# Gather tensors from all GPUs
gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
dist.all_gather(gathered, tensor)

# Each GPU now has all GPUs' tensors
```

### Point-to-Point

```python
# Send from GPU 0 to GPU 1
if rank == 0:
    dist.send(tensor, dst=1)
elif rank == 1:
    dist.recv(tensor, src=0)
```

## NCCL vs RCCL

```
NCCL (NVIDIA): Optimized collective communications for NVIDIA GPUs
RCCL (AMD):    ROCm Collective Communications Library

Both provide:
- AllReduce, AllGather, ReduceScatter
- Broadcast, Reduce
- Send, Receive

Usage:
export NCCL_DEBUG=INFO  # Debug NCCL
export RCCL_DEBUG=INFO  # Debug RCCL
```

## Memory Management

### Unified Memory Across GPUs

```python
# CUDA Unified Memory
tensor = torch.zeros(size, device='cuda:0')
tensor = tensor.cuda(1)  # Migration needed

# Manual pinning for multi-GPU
tensor = torch.zeros(size).pin_memory()
tensor_gpu0 = tensor.cuda(0, non_blocking=True)
tensor_gpu1 = tensor.cuda(1, non_blocking=True)
```

### Memory-Efficient Inference

```python
# Offload unused layers to CPU
class MemoryEfficientModel:
    def __init__(self, model):
        self.layers = model.layers
        self.active_device = 'cuda:0'
        self.offload_device = 'cpu'
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Load layer to GPU
            layer = layer.to(self.active_device)
            x = layer(x)
            
            # Offload processed layer
            layer = layer.to(self.offload_device)
            torch.cuda.empty_cache()
        
        return x
```

## Latency vs Throughput Optimization

### Latency-Optimized (Tensor Parallel)

```
Single request: Split across GPUs
                 
GPU 0 ──┐
GPU 1 ──┼── Result
GPU 2 ──┤
GPU 3 ──┘

Latency: ~Tsingle / N + Communication
Good for: Real-time applications
```

### Throughput-Optimized (Data Parallel)

```
Multiple requests: Each GPU handles subset
                 
Request A → GPU 0 → Result A
Request B → GPU 1 → Result B
Request C → GPU 2 → Result C
Request D → GPU 3 → Result D

Throughput: ~N × Tsingle
Good for: Batch processing
```

## Configuration Examples

### 2-GPU Tensor Parallel

```python
# vLLM tensor parallel
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,  # Split across 2 GPUs
    gpu_memory_utilization=0.9,
)
```

### 4-GPU Pipeline Parallel

```python
# DeepSpeed inference
import deepspeed

model = deepspeed.init_inference(
    model,
    mp_size=4,  # 4-way model parallel
    dtype=torch.float16,
    replace_with_kernel_inject=True,
)
```

## Debugging Multi-GPU

```python
# Check GPU utilization
import subprocess

def gpu_utilization():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return [int(x.strip().replace('%', '')) for x in result.stdout.strip().split('\n')]

# Monitor during inference
utils = []
for batch in data_loader:
    output = model(batch)
    utils.append(gpu_utilization())

print(f"Average GPU utilization: {[sum(u)/len(u) for u in zip(*utils)]}")
```

## Best Practices

1. **Choose right parallelism**: Tensor for latency, Data for throughput
2. **Minimize communication**: Larger chunks, fewer syncs
3. **Balance workload**: Equal work per GPU
4. **Use NVLink/IF**: Faster than PCIe
5. **Profile communication**: Often the bottleneck
6. **Consider memory**: Tensor parallel saves per-GPU memory
