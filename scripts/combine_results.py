import glob
import os
import pandas as pd


def main():
    files = glob.glob("results/*_benchmark.csv")

    if not files:
        print("No benchmark CSV files found.")
        return

    dfs = []

    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = os.path.basename(file)
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)

    output_path = "results/summary.csv"
    result.to_csv(output_path, index=False)

    print(result.to_string(index=False))
    print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()