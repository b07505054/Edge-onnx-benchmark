import os
import pandas as pd


def main():
    summary_path = "results/summary.csv"

    if not os.path.exists(summary_path):
        raise FileNotFoundError("results/summary.csv not found. Run combine_results.py first.")

    summary = pd.read_csv(summary_path)

    pytorch = summary[summary["backend"] == "PyTorch"].sort_values("avg_latency_ms").iloc[0]
    onnx_fp32 = summary[
        (summary["backend"] == "ONNX Runtime") &
        (summary["source_file"].str.contains("mobilenet_v2_fp32"))
    ].sort_values("avg_latency_ms").iloc[0]

    optimized = summary[
        (summary["backend"] == "ONNX Runtime") &
        (summary["source_file"].str.contains("mobilenet_v2_optimized"))
    ].sort_values("avg_latency_ms").iloc[0]

    int8 = summary[
        (summary["backend"] == "ONNX Runtime") &
        (summary["source_file"].str.contains("mobilenet_v2_int8"))
    ].sort_values("avg_latency_ms").iloc[0]

    speedup_vs_pytorch = pytorch["avg_latency_ms"] / optimized["avg_latency_ms"]
    size_reduction = (1 - int8["model_size_mb"] / optimized["model_size_mb"]) * 100
    int8_latency_regression = int8["avg_latency_ms"] / optimized["avg_latency_ms"]

    report = f"""# Edge ONNX Runtime Optimization & Profiling Report

## Overview

This project benchmarks an edge-oriented inference optimization pipeline using MobileNetV2 across PyTorch, ONNX Runtime FP32, optimized ONNX Runtime FP32, and INT8-quantized ONNX Runtime.

The goal is to evaluate latency, throughput, model size, memory usage, and output consistency under CPU-based edge deployment constraints.

## Benchmark Summary

| Model | Backend | Threads | Avg Latency (ms) | P95 Latency (ms) | Throughput (QPS) | Model Size (MB) |
|---|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | PyTorch | N/A | {pytorch["avg_latency_ms"]:.2f} | {pytorch["p95_latency_ms"]:.2f} | {pytorch["throughput_qps"]:.2f} | N/A |
| ONNX FP32 | ONNX Runtime | {int(onnx_fp32["threads"])} | {onnx_fp32["avg_latency_ms"]:.2f} | {onnx_fp32["p95_latency_ms"]:.2f} | {onnx_fp32["throughput_qps"]:.2f} | {onnx_fp32["model_size_mb"]:.2f} |
| Optimized ONNX FP32 | ONNX Runtime | {int(optimized["threads"])} | {optimized["avg_latency_ms"]:.2f} | {optimized["p95_latency_ms"]:.2f} | {optimized["throughput_qps"]:.2f} | {optimized["model_size_mb"]:.2f} |
| INT8 ONNX | ONNX Runtime | {int(int8["threads"])} | {int8["avg_latency_ms"]:.2f} | {int8["p95_latency_ms"]:.2f} | {int8["throughput_qps"]:.2f} | {int8["model_size_mb"]:.2f} |

## Key Findings

- Optimized ONNX Runtime achieved **{speedup_vs_pytorch:.2f}x speedup** over the PyTorch FP32 baseline.
- ONNX Runtime improved throughput from **{pytorch["throughput_qps"]:.2f} QPS** to **{optimized["throughput_qps"]:.2f} QPS** under the best optimized FP32 configuration.
- INT8 quantization reduced model size by **{size_reduction:.1f}%**, from **{optimized["model_size_mb"]:.2f} MB** to **{int8["model_size_mb"]:.2f} MB**.
- However, INT8 introduced a **{int8_latency_regression:.1f}x latency regression** compared with optimized FP32 on the CPU backend.

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
"""

    output_path = "results/report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()