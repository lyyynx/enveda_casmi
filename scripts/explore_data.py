"""Read and take a first look at the Enveda CASMI26 training/test data."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    print(f"Data directory: {DATA_DIR}\n")

    for name in ("train.parquet", "test.parquet"):
        print("=" * 80)
        print(f"  {name}")
        print("=" * 80)
        df = pd.read_parquet(DATA_DIR / name)
        print(f"Shape: {df.shape}")
        print("\nColumn dtypes:")
        print(df.dtypes)
        print("\nFirst rows:")
        print(df.head())
        print("\nMissing values per column:")
        print(df.isna().sum())
        print()

    print("=" * 80)
    print("  sample_submission.csv")
    print("=" * 80)
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print(f"Shape: {sample_sub.shape}")
    print("\nFirst rows:")
    print(sample_sub.head())


if __name__ == "__main__":
    main()
