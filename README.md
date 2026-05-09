# Edge AI Inference Optimization
## ExecuTorch + ONNX Runtime + TFLite Runtime Systems Analysis

Production-style edge inference evaluation pipeline for analyzing runtime behavior, deployment trade-offs, multi-thread scaling, quantization impact, and hardware-aware inference across modern edge AI runtimes.

The project evaluates:
- ONNX Runtime
- ExecuTorch Python runtime
- ExecuTorch C++ runtime
- TFLite XNNPACK delegate

with a focus on:
- Runtime systems engineering
- Edge deployment analysis
- CPU scaling behavior
- Quantization trade-offs
- Performance bottleneck analysis
- Roofline-style systems interpretation

---

## System Overview

![Edge Inference Architecture](results/edge_inference_architecture.png)

This project evaluates edge inference across ONNX Runtime, ExecuTorch, and TFLite, then analyzes:
- latency
- throughput
- CPU utilization
- memory footprint
- quantization behavior
- runtime bottlenecks

under CPU-based edge deployment environments.

---

## Key Capabilities

- Multi-runtime edge inference benchmarking
- ExecuTorch C++ runtime integration
- XNNPACK backend execution
- CPU thread-scaling analysis
- Runtime latency / throughput profiling
- Quantization deployment analysis
- Memory footprint evaluation
- Runtime bottleneck analysis
- Roofline-style performance interpretation
- Hardware-aware inference evaluation

---

## Engineering Motivation

Most ML projects stop at model training or inference APIs.

This project focuses on the systems layer of edge AI deployment:
- runtime execution
- hardware-aware inference
- deployment trade-offs
- scaling behavior
- memory bottlenecks
- runtime profiling

The goal is to understand how inference systems behave under realistic deployment constraints.

---

## Runtime Comparison

This project benchmarks MobileNetV2 across ONNX Runtime, ExecuTorch, and TFLite.

![Runtime Comparison](results/runtime_comparison_final.png)

| Runtime | Backend | Avg Latency |
|---|---|---:|
| ONNX Runtime | CPU EP | ~5.00 ms |
| ExecuTorch Python | Portable / XNNPACK | ~6.00 ms |
| ExecuTorch C++ | XNNPACK, 4 threads | ~5.70 ms |
| TFLite | XNNPACK CPU delegate | ~9.69 ms |

Key insight:
ExecuTorch C++ reaches latency comparable to ONNX Runtime when XNNPACK and multi-threading are enabled, while TFLite provides a mobile-oriented runtime baseline.

---

## Thread Scaling Analysis

### Latency vs Threads
![Thread Scaling](results/executorch_thread_scaling.png)

### CPU Utilization vs Threads
![CPU Usage](results/cpu_usage_scaling.png)

### Throughput Scaling
![Throughput](results/throughput_scaling.png)

### Speedup Analysis
![Speedup](results/speedup_scaling.png)

---

## Key Findings

### 1. Thread Scaling

- Latency improves significantly from 1 → 4 threads
- ~2.5× speedup achieved
- Diminishing returns beyond 4 threads

This suggests:
- effective parallelization initially
- saturation near hardware limits

---

### 2. CPU Utilization

Observed CPU utilization:
- ~25% (1 thread)
- ~90% (8 threads)

This confirms:
- workload is CPU-intensive
- parallel execution is active
- thread scheduling meaningfully affects performance

---

### 3. Throughput Scaling

Throughput scales nearly linearly up to 4 threads before plateauing.

This indicates:
- efficient parallel execution
- emerging memory/system bottlenecks at higher thread counts

---

### 4. Runtime Comparison

ONNX Runtime achieved the lowest observed latency under the current Windows CPU setup.

ExecuTorch C++ reached comparable performance using:
- XNNPACK backend
- multi-thread execution
- C++ runtime integration

ExecuTorch Python demonstrated modest runtime overhead relative to the C++ path.

TFLite successfully executed using the XNNPACK CPU delegate, providing a mobile-oriented runtime baseline.

---

### 5. Quantization Analysis

Quantization reduced:
- runtime memory footprint
- deployment artifact size

However:
- lower memory usage did not consistently improve latency

This highlights deployment trade-offs between:
- memory efficiency
- runtime execution overhead

---

## Systems Engineering Insights

Observed runtime behavior suggests:

- Near-linear scaling from 1 → 4 threads
- Saturation beyond 4 threads
- Increasing CPU utilization with thread count
- Diminishing returns likely caused by:
  - memory bandwidth pressure
  - scheduling overhead
  - synchronization cost

Quantization reduced runtime memory footprint, but did not consistently improve latency, highlighting deployment trade-offs between memory efficiency and execution overhead.

These behaviors are consistent with hardware-aware edge inference systems operating near CPU resource limits.

---

## Roofline-Style Interpretation

The observed scaling behavior is consistent with Roofline-style performance analysis:

- Performance improves with increased parallelism in the compute-bound region
- Beyond 4 threads, scaling efficiency declines
- This suggests the workload approaches hardware constraints such as:
  - memory bandwidth limits
  - synchronization overhead
  - shared-resource contention

This aligns with the transition from compute-bound execution toward memory-bound behavior.

---

## Memory and Deployment Analysis

The project also evaluates:
- runtime RSS memory usage
- deployment artifact size
- model loading overhead

Observed behavior demonstrates:

```text
Artifact size ≠ runtime memory footprint
```

Runtime memory behavior depends on:
- tensor allocation
- runtime buffers
- intermediate activations
- operator workspace requirements

not only serialized model size.

---

## Methodology

### Model
- MobileNetV2

### Execution Environment
- CPU-only inference
- Windows execution environment
- XNNPACK-enabled runtimes

### Metrics Collected
- Average latency
- Tail latency (p95 / p99)
- Throughput
- CPU utilization
- Speedup scaling
- Runtime memory usage
- Model load latency

### Benchmarking Controls
- Fixed number of executions
- Warmup handling
- Thread-count variation
- Repeated execution runs

---

## Tech Stack

### Languages
- Python
- C++

### Inference Runtimes
- ONNX Runtime
- ExecuTorch
- TFLite

### Backends
- XNNPACK
- CPU Execution Provider

### Libraries / Tooling
- PyTorch
- TensorFlow Lite
- Matplotlib
- NumPy

---

## Project Structure

```text
edge-onnx-benchmark/
├── benchmarks/        # benchmarking + plotting scripts
├── results/           # generated metrics and visualizations
├── executorch_export/
├── models/
├── docs/
└── README.md
```

---

## What This Demonstrates

This project demonstrates ability to:

- Deploy models across multiple runtimes
- Work with low-level inference runtimes (C++)
- Analyze thread-level scaling behavior
- Perform runtime bottleneck analysis
- Benchmark deployment trade-offs
- Evaluate quantization impact
- Interpret hardware-aware inference behavior
- Design reproducible performance engineering workflows

---

## Future Extensions

- Android / embedded deployment
- Apple Silicon benchmarking
- TFLite GPU / NPU delegate evaluation
- CoreML / MLX integration
- Hardware-counter profiling
- Thermal / power analysis
- Delegate partitioning analysis
- Real roofline modeling
- Memory allocator analysis

---

## References

- ONNX Runtime
- ExecuTorch
- TensorFlow Lite
- XNNPACK
- PyTorch