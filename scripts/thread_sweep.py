import subprocess
import pandas as pd
import os

THREADS = [1, 2, 4, 8]
MODEL_PATH = "models/mobilenet_v2_optimized.onnx"

def main():
    os.makedirs("results", exist_ok=True)

    rows = []

    for t in THREADS:
        print(f"Running ONNX benchmark with {t} threads...")

        subprocess.run([
            "python",
            "scripts/benchmark.py",
            "--backend", "onnx",
            "--model-path", MODEL_PATH,
            "--batch-size", "1",
            "--threads", str(t),
            "--iterations", "100",
        ], check=True)

        csv_path = f"results/mobilenet_v2_optimized_bs1_t{t}_benchmark.csv"
        df = pd.read_csv(csv_path)
        rows.append(df.iloc[0].to_dict())

    result = pd.DataFrame(rows)
    output_path = "results/thread_sweep.csv"
    result.to_csv(output_path, index=False)

    print(result.to_string(index=False))
    print(f"\nSaved thread sweep to {output_path}")

if __name__ == "__main__":
    main()