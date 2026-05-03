import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    input_path = "results/thread_sweep.csv"
    output_path = "results/thread_latency_curve.png"

    df = pd.read_csv(input_path)

    plt.figure()
    plt.plot(df["threads"], df["avg_latency_ms"], marker="o")
    plt.xlabel("Number of CPU Threads")
    plt.ylabel("Average Latency (ms)")
    plt.title("ONNX Runtime Thread Scaling - MobileNetV2")
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()