# Chapter 9: Batching Strategies

## Why Batching Matters

Batching amortizes fixed costs and improves GPU utilization.

```
Single Request:
┌─────────┐
│ Request │ → GPU (5% utilized) → Response
└─────────┘

Batched Requests:
┌─────────┐
│ Req 1   │
│ Req 2   │ → GPU (80% utilized) → Responses
│ Req 3   │
│ Req 4   │
└─────────┘
```

## Static Batching

### Fixed Batch Size

```python
# Simple static batching
BATCH_SIZE = 32

def process_batch(requests):
    # Pad to fixed size
    inputs = pad_to_batch_size(requests, BATCH_SIZE)
    
    # Run inference
    outputs = model(inputs)
    
    # Return only valid outputs
    return outputs[:len(requests)]
```

### Pros and Cons

**Pros:**
- Simple to implement
- Predictable memory usage
- Easy to optimize kernels

**Cons:**
- Wastes compute on padding
- Fixed latency regardless of load
- Suboptimal for variable-length inputs

## Dynamic Batching

### Request Accumulation

```python
import asyncio
from collections import deque

class DynamicBatcher:
    def __init__(self, max_batch_size=32, max_wait_ms=10):
        self.queue = deque()
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        
    async def add_request(self, request):
        future = asyncio.Future()
        self.queue.append((request, future))
        
        # Start batch timer on first request
        if len(self.queue) == 1:
            asyncio.create_task(self._process_after_wait())
        
        return await future
    
    async def _process_after_wait(self):
        await asyncio.sleep(self.max_wait)
        await self._process_batch()
    
    async def _process_batch(self):
        batch_requests = []
        batch_futures = []
        
        while self.queue and len(batch_requests) < self.max_batch_size:
            req, future = self.queue.popleft()
            batch_requests.append(req)
            batch_futures.append(future)
        
        # Run batched inference
        results = model(batch_requests)
        
        # Distribute results
        for future, result in zip(batch_futures, results):
            future.set_result(result)
```

### Configuration Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| max_batch_size | Maximum requests per batch | 8-128 |
| max_wait_ms | Maximum wait time for batching | 5-50 ms |
| bucket_sizes | Sequence length buckets | [32, 64, 128, 256] |

## Continuous Batching (LLM-Specific)

### The Problem with Static Batching for LLMs

```
Request 1: "Hello" → 10 tokens
Request 2: "Write essay" → 500 tokens
Request 3: "Hi" → 5 tokens

Static batch: All wait for longest (500 tokens)
Continuous: Short requests finish and new ones join
```

### Continuous Batching Implementation

```python
class ContinuousBatcher:
    def __init__(self, model, max_batch_tokens=4096):
        self.model = model
        self.max_batch_tokens = max_batch_tokens
        self.active_sequences = []
        self.waiting_queue = []
        
    def step(self):
        # Add new sequences if capacity allows
        while self.waiting_queue:
            seq = self.waiting_queue[0]
            current_tokens = sum(s.current_length for s in self.active_sequences)
            
            if current_tokens + seq.current_length <= self.max_batch_tokens:
                self.active_sequences.append(self.waiting_queue.pop(0))
            else:
                break
        
        if not self.active_sequences:
            return
        
        # Run one decode step for all active sequences
        # Sequences can have different lengths!
        outputs = self.model.decode_step(self.active_sequences)
        
        # Remove completed sequences
        completed = []
        for seq, output in zip(self.active_sequences, outputs):
            seq.append_token(output)
            if seq.is_complete:
                completed.append(seq)
        
        for seq in completed:
            self.active_sequences.remove(seq)
```

### vLLM PagedAttention

```
Traditional: Pre-allocate max_length KV cache per sequence
PagedAttention: Allocate KV cache pages on demand

Memory efficiency: 2-4x improvement
Enables larger batches → higher throughput
```

## Sequence Length Bucketing

### Why Buckets?

```
Without buckets:
Seq lengths: [10, 100, 15, 95]
Padded to max: [100, 100, 100, 100]  // 78% waste

With buckets [32, 64, 128]:
[10, 100, 15, 95] → Bucket 32: [10, 15]
                  → Bucket 128: [100, 95]
                  // Much less padding
```

### Implementation

```python
def bucket_sequences(sequences, bucket_sizes=[32, 64, 128, 256, 512]):
    buckets = {size: [] for size in bucket_sizes}
    
    for seq in sequences:
        # Find smallest bucket that fits
        for size in bucket_sizes:
            if seq.length <= size:
                buckets[size].append(seq)
                break
        else:
            # Handle sequences longer than max bucket
            buckets[max(bucket_sizes)].append(seq)
    
    return buckets

def process_bucketed(sequences):
    buckets = bucket_sequences(sequences)
    results = []
    
    for bucket_size, bucket_seqs in buckets.items():
        if bucket_seqs:
            padded = pad_sequences(bucket_seqs, bucket_size)
            outputs = model(padded)
            results.extend(zip(bucket_seqs, outputs))
    
    return results
```

## Batching for Different Workloads

### Computer Vision

```python
# Image classification: Simple static batching
# All inputs same size (224x224)
batch = torch.stack(images)  # [B, 3, 224, 224]
outputs = model(batch)

# Object detection: Varying objects per image
# Use padding or list processing
batch = nested_tensor(images)  # Handle varying sizes
outputs, boxes = model(batch)
```

### NLP/Transformers

```python
# Use attention masks for variable lengths
batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
# batch contains input_ids, attention_mask

outputs = model(**batch)
```

### Recommendation Systems

```python
# Feature batching for embedding lookups
user_features = batch_lookup(user_ids)     # Dense batch
item_features = batch_lookup(item_ids)     # Dense batch
sparse_features = sparse_batch(features)   # Ragged batch

outputs = model(user_features, item_features, sparse_features)
```

## Performance Tuning

### Finding Optimal Batch Size

```python
import time
import torch

def find_optimal_batch_size(model, input_shape, max_batch=256):
    results = []
    
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if batch_size > max_batch:
            break
            
        try:
            input_data = torch.randn(batch_size, *input_shape).cuda()
            
            # Warmup
            for _ in range(10):
                _ = model(input_data)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(100):
                _ = model(input_data)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            throughput = (batch_size * 100) / elapsed
            latency = (elapsed * 1000) / 100  # ms per batch
            
            results.append({
                'batch_size': batch_size,
                'throughput': throughput,
                'latency_ms': latency,
                'latency_per_sample': latency / batch_size
            })
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                break
            raise
    
    return results
```

### Batch Size vs. Latency Tradeoff

```
                  Throughput
                     ↑
                     │      ┌────────── Throughput curve
                     │     /
                     │    /
                     │   /
                     │  /
                     │ /
                     │/________________→ Batch Size
                     
                  Latency
                     ↑
                     │                  Latency curve
                     │                /
                     │              /
                     │            /
                     │          /
                     │________/________→ Batch Size

Sweet spot: Maximum throughput within latency budget
```

## Best Practices

1. **Profile your workload**: Understand request patterns
2. **Use dynamic batching**: Better than fixed for variable loads
3. **Consider continuous batching**: Essential for LLM inference
4. **Bucket by sequence length**: Reduce padding waste
5. **Set appropriate timeouts**: Balance latency vs. efficiency
6. **Monitor queue depths**: Detect bottlenecks early
