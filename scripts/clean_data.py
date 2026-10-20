"""Apply spectrum cleaning to train and test data and save cleaned copies.

Usage:
    uv run python scripts/clean_data.py                     # full run, saves cleaned parquet
    uv run python scripts/clean_data.py --sample 10000      # quick trial run
    uv run python scripts/clean_data.py --floor 0.01 --top-k-per-window 0 --max-peaks 128
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from enveda_casmi26.cleaning import CleanConfig, SpectrumCleaner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def build_config(args: argparse.Namespace) -> CleanConfig:
    return CleanConfig(
        relative_intensity_floor=args.floor,
        precursor_margin_da=args.precursor_margin,
        deisotope=not args.no_deisotope,
        isotope_tolerance_da=args.isotope_tol,
        top_k_per_window=args.top_k_per_window or None,
        window_da=args.window_da,
        max_peaks=args.max_peaks or None,
        min_peaks=args.min_peaks,
        intensity_transform=args.transform,
    )


def clean_file(cleaner: SpectrumCleaner, name: str, sample: int | None, save: bool) -> None:
    path = DATA_DIR / f"{name}.parquet"
    out_path = DATA_DIR / f"{name}_clean.parquet"
    print(f"\nLoading {name} ...")
    df = pd.read_parquet(path)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)

    t0 = time.time()
    cleaned = cleaner.clean_dataframe(df)
    print(f"  cleaning took {time.time() - t0:.1f}s")
    cleaner.summarize(cleaned, name)

    if save:
        cleaned.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  saved {out_path.name} ({size_mb:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--floor", type=float, default=0.001, help="relative intensity floor (0 disables)")
    parser.add_argument("--precursor-margin", type=float, default=1.5, help="Da above precursor to keep")
    parser.add_argument("--no-deisotope", action="store_true")
    parser.add_argument("--isotope-tol", type=float, default=0.01)
    parser.add_argument("--top-k-per-window", type=int, default=6, help="0 disables window selection")
    parser.add_argument("--window-da", type=float, default=50.0)
    parser.add_argument("--max-peaks", type=int, default=128, help="global cap, 0 disables")
    parser.add_argument("--min-peaks", type=int, default=5, help="flag threshold (rows are kept)")
    parser.add_argument("--transform", choices=["sqrt", "log", "none"], default="sqrt")
    parser.add_argument("--sample", type=int, default=None, help="only clean a sample of train")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    print(f"CleanConfig: {config}")

    cleaner = SpectrumCleaner(config)
    if not args.skip_test:
        clean_file(cleaner, "test", None, not args.no_save)
    if not args.skip_train:
        clean_file(cleaner, "train", args.sample, not args.no_save)


if __name__ == "__main__":
    main()
