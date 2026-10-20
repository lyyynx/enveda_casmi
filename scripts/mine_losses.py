"""Tier 3: mine recurring fragmentation patterns from unexplained peaks.

For sampled training spectra, peaks unexplained by the default (discrimination-
tuned) configuration are re-matched with a permissive discovery configuration.
Recurring (kind, neutral loss, h-shift) patterns weighted by peak intensity
indicate candidate rules for extending the fragmenter.

Usage:
    uv run python scripts/mine_losses.py [--n-spectra 300]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from enveda_casmi26.fragmentation import ADDUCTS, ADDUCT_ALIASES, LOSS_DEFS, Fragmentor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BENCH_DIR = PROJECT_ROOT / "benchmark"
SUPPORTED_ADDUCTS = set(ADDUCTS) | set(ADDUCT_ALIASES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-spectra", type=int, default=300)
    parser.add_argument("--min-peaks", type=int, default=10)
    parser.add_argument("--tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("Loading cleaned training data ...")
    df = pd.read_parquet(DATA_DIR / "train_clean.parquet")
    df = df[
        (df["drop_reason"] == "")
        & (df["num_peaks_clean"] >= args.min_peaks)
        & (df["adduct"].isin(SUPPORTED_ADDUCTS))
        & df["inchikey14"].notna()
    ].drop_duplicates(subset="inchikey14", keep="first")
    if len(df) > args.n_spectra:
        df = df.sample(args.n_spectra, random_state=args.seed)
    print(f"mining {len(df)} spectra\n")

    default_frag = None
    patterns: list[dict] = []
    total_unexplained_intensity = 0.0
    total_explained_intensity = 0.0
    t0 = time.time()
    for i, spec in enumerate(df.itertuples(index=False)):
        try:
            default_frag = Fragmentor(spec.normalized_smiles, adduct=spec.adduct,
                                       tolerance_ppm=args.tolerance_ppm, include_precursor_losses=False)
            discovery_frag = Fragmentor(
                spec.normalized_smiles, adduct=spec.adduct,
                tolerance_ppm=args.tolerance_ppm, include_precursor_losses=False,
                max_serial_losses=3, min_mz=25.0, neutral_losses=tuple(LOSS_DEFS),
                h_shifts=(-3, -2, -1, 0, 1, 2, 3),
            )
        except Exception:
            continue
        mzs = np.asarray(spec.ms2_mzs, dtype=float)
        ints = np.asarray(spec.ms2_normalized_intensities, dtype=float)

        base = default_frag.match_peaks(mzs, ints)
        unexplained_mask = np.ones(len(mzs), dtype=bool)
        if not base.empty:
            unexplained_mask[base["peak_index"].to_numpy()] = False
        unexp_mzs, unexp_ints = mzs[unexplained_mask], ints[unexplained_mask]
        total_unexplained_intensity += float(unexp_ints.sum())
        if len(unexp_mzs) == 0:
            continue

        discovered = discovery_frag.match_peaks(unexp_mzs, unexp_ints)
        if discovered.empty:
            continue
        total_explained_intensity += float(discovered["intensity"].sum())
        for row in discovered.itertuples(index=False):
            patterns.append({
                "kind": row.kind,
                "neutral_loss": row.neutral_loss,
                "h_shift": row.h_shift,
                "peak_intensity": row.intensity,
                "abs_ppm": abs(row.error_ppm),
                "frag_heavy_atoms": row.frag_heavy_atoms,
            })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)} spectra done ({time.time() - t0:.0f}s)")

    print(f"\nunexplained intensity by default config: {total_unexplained_intensity:.1f}")
    print(f"re-explained by discovery config:         {total_explained_intensity:.1f} "
          f"({total_explained_intensity / max(total_unexplained_intensity, 1e-9):.1%})")

    mined = pd.DataFrame(patterns)
    BENCH_DIR.mkdir(exist_ok=True)
    out = BENCH_DIR / "mined_patterns.csv"
    mined.to_csv(out, index=False)

    agg = (
        mined.groupby(["kind", "neutral_loss", "h_shift"])
        .agg(n_peaks=("peak_intensity", "size"),
             total_intensity=("peak_intensity", "sum"),
             mean_abs_ppm=("abs_ppm", "mean"),
             mean_frag_size=("frag_heavy_atoms", "mean"))
        .sort_values("total_intensity", ascending=False)
    )
    print(f"\ntop mined patterns (by intensity):")
    print(agg.head(25).round(2).to_string())
    print(f"\npattern table saved to {out}")


if __name__ == "__main__":
    main()
