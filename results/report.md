# Edge ONNX Runtime Optimization & Profiling Report

## Overview

This project benchmarks an edge-oriented inference optimization pipeline using MobileNetV2 across PyTorch, ONNX Runtime FP32, optimized ONNX Runtime FP32, and INT8-quantized ONNX Runtime.

The goal is to evaluate latency, throughput, model size, memory usage, and output consistency under CPU-based edge deployment constraints.

## Benchmark Summary

| Model | Backend | Threads | Avg Latency (ms) | P95 Latency (ms) | Throughput (QPS) | Model Size (MB) |
|---|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | PyTorch | N/A | 26.12 | 32.31 | 38.29 | N/A |
| ONNX FP32 | ONNX Runtime | 1 | 9.99 | 11.02 | 100.10 | 0.27 |
| Optimized ONNX FP32 | ONNX Runtime | 4 | 4.67 | 5.78 | 214.35 | 13.33 |
| INT8 ONNX | ONNX Runtime | 4 | 105.48 | 116.02 | 9.48 | 3.45 |

## Key Findings

- Optimized ONNX Runtime achieved **5.60x speedup** over the PyTorch FP32 baseline.
- ONNX Runtime improved throughput from **38.29 QPS** to **214.35 QPS** under the best optimized FP32 configuration.
- INT8 quantization reduced model size by **74.1%**, from **13.33 MB** to **3.45 MB**.
- However, INT8 introduced a **22.6x latency regression** compared with optimized FP32 on the CPU backend.

## Accuracy / Consistency Validation

| Model | Top-1 Consistency with PyTorch |
|---|---:|
| ONNX FP32 | 100% |
| Optimized ONNX FP32 | 100% |
| INT8 ONNX | 3% |

## Analysis

The optimized ONNX FP32 model preserved output consistency while significantly improving latency and throughput. This makes it the best deployment candidate under the current CPU execution setting.

The INT8 model achieved a substantial model-size reduction, but both latency and output consistency degraded significantly. This suggests that naive dynamic quantization is unsuitable for this MobileNetV2 convolution-heavy workload without calibration-aware static quantization or backend-specific acceleration.

## Next Steps

- Add static quantization with calibration data.
- Add C++ ONNX Runtime inference demo with CMake.
- Add profiling by batch size and thread count.
- Extend benchmark to transformer models where dynamic quantization is more effective.
