"""Profile MS2 spectra to choose cleaning thresholds.

Usage:
    uv run python scripts/profile_spectra.py [--sample 200000]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def profile(df: pd.DataFrame, name: str) -> dict:
    n_spec = len(df)
    if "num_peaks" not in df.columns:
        df = df.copy()
        df["num_peaks"] = [len(m) for m in df["ms2_mzs"]]
    stats = {
        "spectra": n_spec,
        "peaks_total": float(df["num_peaks"].sum()),
        "peaks_per_spectrum": {
            "min": int(df["num_peaks"].min()),
            "p25": float(df["num_peaks"].quantile(0.25)),
            "median": float(df["num_peaks"].quantile(0.50)),
            "p75": float(df["num_peaks"].quantile(0.75)),
            "p95": float(df["num_peaks"].quantile(0.95)),
            "p99": float(df["num_peaks"].quantile(0.99)),
            "max": int(df["num_peaks"].max()),
        },
        "frac_spectra_lt5peaks": float((df["num_peaks"] < 5).mean()),
        "frac_spectra_gt128peaks": float((df["num_peaks"] > 128).mean()),
        "frac_spectra_gt512peaks": float((df["num_peaks"] > 512).mean()),
    }

    max_intensity = np.zeros(n_spec)
    n_below = {f: np.zeros(n_spec, dtype=np.int64) for f in (0.001, 0.002, 0.005, 0.01, 0.02)}
    n_above = np.zeros(n_spec, dtype=np.int64)
    n_isotope = np.zeros(n_spec, dtype=np.int64)

    for i, (mzs, ints, prec) in enumerate(
        zip(df["ms2_mzs"], df["ms2_normalized_intensities"], df["precursor_mz"])
    ):
        mzs = np.asarray(mzs, dtype=float)
        ints = np.asarray(ints, dtype=float)
        max_intensity[i] = ints.max() if len(ints) else np.nan
        for f in n_below:
            n_below[f][i] = (ints < f).sum()
        if np.isfinite(prec):
            n_above[i] = (mzs > prec + 1.5).sum()
        order = np.argsort(mzs)
        ms, ins = mzs[order], ints[order]
        lo = np.searchsorted(ms, ms + 1.00335 - 0.01)
        hi = np.searchsorted(ms, ms + 1.00335 + 0.01)
        cnt = 0
        for a in range(len(ms)):
            for b in range(lo[a], hi[a]):
                if ins[b] <= ins[a]:
                    cnt += 1
        n_isotope[i] = cnt

    stats["max_intensity"] = {"min": float(np.nanmin(max_intensity)), "max": float(np.nanmax(max_intensity))}
    total_peaks = stats["peaks_total"]
    stats["peaks_below_floor_frac"] = {str(f): float(n_below[f].sum() / total_peaks) for f in n_below}
    stats["peaks_above_precursor_frac"] = float(n_above.sum() / total_peaks)
    stats["spectra_with_peaks_above_precursor_frac"] = float((n_above > 0).mean())
    stats["isotope_companions_frac_of_peaks"] = float(n_isotope.sum() / total_peaks)
    stats["spectra_with_isotopes_frac"] = float((n_isotope > 0).mean())

    print(f"\n===== {name} =====")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200000)
    args = parser.parse_args()

    cols = ["ms2_mzs", "ms2_normalized_intensities", "num_peaks", "precursor_mz"]
    print("Loading test ...")
    test = pd.read_parquet(DATA_DIR / "test.parquet", columns=cols)
    profile(test, "test (full)")

    print("\nLoading train sample ...")
    train = pd.read_parquet(DATA_DIR / "train.parquet", columns=cols)
    if args.sample and len(train) > args.sample:
        train = train.sample(args.sample, random_state=42)
    profile(train, f"train (sample of {len(train)})")


if __name__ == "__main__":
    main()
