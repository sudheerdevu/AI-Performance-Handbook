# AI Performance Handbook 📖

A comprehensive guide to AI/ML performance engineering, covering everything from hardware fundamentals to production optimization strategies.

## Overview

This handbook distills years of AI performance engineering experience into actionable guidance. Whether you're optimizing inference latency for real-time applications or maximizing training throughput, you'll find practical techniques here.

## Table of Contents

### Part I: Foundations
1. [GPU Architecture Fundamentals](docs/01-gpu-architecture.md)
2. [Memory Hierarchy Deep Dive](docs/02-memory-hierarchy.md)
3. [Understanding Compute vs Memory Bound](docs/03-compute-vs-memory.md)

### Part II: Profiling & Analysis
4. [Profiling Tools Overview](docs/04-profiling-tools.md)
5. [Interpreting Hardware Counters](docs/05-hardware-counters.md)
6. [Bottleneck Classification](docs/06-bottleneck-classification.md)

### Part III: Model Optimization
7. [Quantization Strategies](docs/07-quantization.md)
8. [Operator Fusion Patterns](docs/08-operator-fusion.md)
9. [Batching Strategies](docs/09-batching.md)

### Part IV: System Optimization
10. [CPU-GPU Data Transfer](docs/10-data-transfer.md)
11. [Multi-GPU Scaling](docs/11-multi-gpu.md)
12. [Production Deployment](docs/12-production.md)

### Appendices
- [Performance Cheat Sheet](docs/appendix-cheatsheet.md)
- [Tool Reference](docs/appendix-tools.md)
- [Glossary](docs/appendix-glossary.md)

## Quick Reference

### Performance Optimization Decision Tree

```
Inference too slow?
├── Profile with rocprof/nsys
├── Is it compute bound?
│   ├── Yes → Check kernel efficiency
│   │   ├── Low occupancy? → Reduce register pressure
│   │   ├── Low IPC? → Check for stalls
│   │   └── Good metrics? → Consider quantization
│   └── No → Memory bound
│       ├── Poor coalescing? → Optimize memory layout
│       ├── Cache misses? → Improve locality
│       └── Bandwidth limited? → Reduce data movement
└── Check if it's ops-limited
    ├── Too many small kernels? → Fuse operations
    └── Launch overhead? → Batch more work
```

### Golden Rules

1. **Measure First**: Never optimize blindly. Profile, then optimize, then measure again.
2. **Amdahl's Law**: Focus on the biggest bottleneck. 10% of 5% is still small.
3. **Memory is King**: In AI workloads, memory bandwidth often limits performance.
4. **Batch When Possible**: Amortize fixed costs across more work.
5. **Know Your Hardware**: Architecture-aware optimization beats generic approaches.

## Target Audience

- ML Engineers optimizing inference deployments
- Research Scientists accelerating training
- Platform Engineers building AI infrastructure
- Students learning GPU programming

## Prerequisites

- Basic understanding of neural networks
- Familiarity with Python and/or C++
- Some exposure to parallel programming concepts

## How to Use This Handbook

1. **New to GPU optimization**: Start with Part I to build foundations
2. **Have a specific problem**: Jump to relevant chapter in Part II-IV
3. **Quick reference**: Use the appendices and cheat sheets

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Author

Sudheer Devu - AI Performance Engineer

## License

CC BY-SA 4.0 - Share and adapt with attribution
