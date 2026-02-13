# Chapter 12: Production Deployment

## From Development to Production

```
Development                    Production
├── Small batches             ├── High throughput
├── Single GPU               ├── Multi-GPU/Node
├── Debugging priority       ├── Reliability priority
├── Flexible                 ├── Optimized
└── Manual monitoring        └── Automated ops
```

## Inference Server Options

### Triton Inference Server

```python
# Model configuration (config.pbtxt)
name: "my_model"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [1000]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0, 1]
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 5000
  preferred_batch_size: [8, 16, 32]
}
```

### vLLM for LLMs

```python
from vllm import LLM, SamplingParams

# Server deployment
llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# OpenAI-compatible API
# vllm serve mistralai/Mistral-7B-v0.1 --port 8000
```

### TensorRT-LLM

```python
# Build optimized engine
import tensorrt_llm
from tensorrt_llm import BuildConfig

build_config = BuildConfig(
    max_batch_size=64,
    max_input_len=1024,
    max_output_len=1024,
    precision='float16',
)

engine = tensorrt_llm.build(model, build_config)
engine.save("./engine")
```

## Containerization

### Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy model and code
COPY model/ /app/model/
COPY src/ /app/src/

# Set environment
ENV MODEL_PATH=/app/model
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "src.server", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
      - name: inference
        image: myregistry/inference:v1
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
```

## Monitoring and Observability

### Metrics to Track

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('inference_latency_seconds', 'Request latency',
                            buckets=[.01, .025, .05, .1, .25, .5, 1, 2.5, 5])
REQUEST_ERRORS = Counter('inference_errors_total', 'Total errors')

# Resource metrics
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu'])
GPU_MEMORY_USED = Gauge('gpu_memory_used_bytes', 'GPU memory used', ['gpu'])
BATCH_SIZE = Histogram('inference_batch_size', 'Batch sizes',
                       buckets=[1, 2, 4, 8, 16, 32, 64])

# Throughput metrics
TOKENS_PER_SECOND = Gauge('tokens_per_second', 'Throughput for LLM')
SAMPLES_PER_SECOND = Gauge('samples_per_second', 'Throughput for batch inference')
```

### GPU Monitoring

```python
import pynvml

def get_gpu_metrics():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    metrics = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        metrics.append({
            'gpu': i,
            'utilization': util.gpu,
            'memory_used': memory.used,
            'memory_total': memory.total,
            'temperature': temp,
        })
    
    return metrics
```

### Logging Best Practices

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        
    def log_inference(self, request_id, latency_ms, batch_size, status):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request_id,
            'event': 'inference',
            'latency_ms': latency_ms,
            'batch_size': batch_size,
            'status': status,
        }
        self.logger.info(json.dumps(log_entry))
```

## Auto-Scaling

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: inference_queue_depth
      target:
        type: AverageValue
        averageValue: "100"
```

### Custom Scaling Logic

```python
class AutoScaler:
    def __init__(self, min_replicas=1, max_replicas=10):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas
        
    def calculate_desired_replicas(self, metrics):
        # P99 latency target: 100ms
        if metrics['p99_latency_ms'] > 150:
            # Scale up aggressively
            return min(self.current_replicas * 2, self.max_replicas)
        elif metrics['p99_latency_ms'] < 50 and metrics['gpu_util'] < 30:
            # Scale down slowly
            return max(self.current_replicas - 1, self.min_replicas)
        return self.current_replicas
```

## Model Updates

### Blue-Green Deployment

```yaml
# Blue (current) deployment
apiVersion: v1
kind: Service
metadata:
  name: inference-blue
spec:
  selector:
    app: inference
    version: v1
---
# Green (new) deployment
apiVersion: v1
kind: Service
metadata:
  name: inference-green
spec:
  selector:
    app: inference
    version: v2
---
# Traffic switch via ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: inference-ingress
spec:
  rules:
  - http:
      paths:
      - path: /inference
        backend:
          service:
            name: inference-green  # Switch from blue to green
            port:
              number: 8000
```

### Canary Deployment

```yaml
# Split traffic between versions
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: inference-canary
spec:
  hosts:
  - inference
  http:
  - route:
    - destination:
        host: inference-v1
      weight: 90
    - destination:
        host: inference-v2
      weight: 10
```

## Error Handling

```python
from fastapi import FastAPI, HTTPException
import asyncio

app = FastAPI()

@app.post("/inference")
async def inference(request: InferenceRequest):
    try:
        # Set timeout
        result = await asyncio.wait_for(
            run_inference(request),
            timeout=30.0
        )
        return result
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
        
    except torch.cuda.OutOfMemoryError:
        # Clear cache and retry with smaller batch
        torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="GPU OOM, retry later")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error")
```

## Production Checklist

- [ ] Model optimized (quantization, fusion)
- [ ] Load testing completed
- [ ] Monitoring dashboards ready
- [ ] Alerting configured
- [ ] Auto-scaling tested
- [ ] Rollback procedure documented
- [ ] Error handling comprehensive
- [ ] Logging structured
- [ ] Health checks implemented
- [ ] Resource limits set
- [ ] Security reviewed
- [ ] Documentation complete
