import time
import numpy as np



def simulate_inference(input_tensor):
    start = time.perf_counter()
    _ = np.dot(input_tensor, input_tensor.T)
    return (time.perf_counter() - start) * 1000


def main():
    runs = 100

    input_tensor = np.random.rand(1, 128).astype(np.float32)

    latencies = []

    for _ in range(runs):
        latency = simulate_inference(input_tensor)
        latencies.append(latency)

    print("=== Inference Benchmark (Simulated) ===")
    print(f"avg: {np.mean(latencies):.4f} ms")
    print(f"p95: {np.percentile(latencies, 95):.4f} ms")
    print(f"p99: {np.percentile(latencies, 99):.4f} ms")


if __name__ == "__main__":
    main()