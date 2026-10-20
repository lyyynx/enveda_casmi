"""Collect a training dataset of fragment-spectrum matches with true/decoy labels.

For each sampled training spectrum, the true structure and up to N
same-formula decoy structures are fragmented with a permissive *discovery*
configuration (full loss dictionary, serial loss chains). Every best-per-peak
match becomes one row labeled is_decoy. Spectra used by the decoy benchmark
(seed-42 sample) are excluded so evaluation stays honest.

Usage:
    uv run python scripts/collect_matches.py [--n-spectra 1000]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from enveda_casmi26.fragmentation import ADDUCTS, ADDUCT_ALIASES, LOSS_DEFS, Fragmentor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUPPORTED_ADDUCTS = set(ADDUCTS) | set(ADDUCT_ALIASES)


def make_fragmentor(smiles: str, adduct: str, tolerance_ppm: float, config: str) -> Fragmentor:
    if config == "discovery":
        return Fragmentor(
            smiles,
            adduct=adduct,
            tolerance_ppm=tolerance_ppm,
            include_precursor_losses=False,
            max_serial_losses=2,
            min_mz=25.0,
            neutral_losses=tuple(LOSS_DEFS),
        )
    if config == "default":
        return Fragmentor(
            smiles,
            adduct=adduct,
            tolerance_ppm=tolerance_ppm,
            include_precursor_losses=False,
        )
    raise ValueError(f"Unknown config: {config}")


def load_eligible(df: pd.DataFrame, min_peaks: int) -> pd.DataFrame:
    return df[
        (df["drop_reason"] == "")
        & (df["num_peaks_clean"] >= min_peaks)
        & (df["adduct"].isin(SUPPORTED_ADDUCTS))
        & df["inchikey14"].notna()
    ].drop_duplicates(subset="inchikey14", keep="first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-spectra", type=int, default=1000)
    parser.add_argument("--n-decoys", type=int, default=9)
    parser.add_argument("--min-peaks", type=int, default=10)
    parser.add_argument("--tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--config", choices=["default", "discovery"], default="default")
    parser.add_argument("--exclude-enveda", action="store_true",
                        help="exclude enveda-180 (placeholder-test library) queries and decoys")
    parser.add_argument("--holdout", default=None,
                        help="parquet with an inchikey14 column of structures to exclude")
    parser.add_argument("--out", default="match_dataset.parquet")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading cleaned training data ...")
    df = pd.read_parquet(DATA_DIR / "train_clean.parquet")
    eligible = load_eligible(df, args.min_peaks)
    if args.exclude_enveda:
        eligible = eligible[eligible["ingest_lib"] != "enveda-180"]
    if args.holdout:
        holdout_keys = set(pd.read_parquet(args.holdout, columns=["inchikey14"])["inchikey14"])
        eligible = eligible[~eligible["inchikey14"].isin(holdout_keys)]
    print(f"eligible structures after exclusions: {len(eligible):,}")

    benchmark_keys = set(
        eligible.sample(min(150, len(eligible)), random_state=42)["inchikey14"]
    )
    pool = eligible[~eligible["inchikey14"].isin(benchmark_keys)]
    print(f"eligible structures: {len(eligible):,} (excluding {len(benchmark_keys)} benchmark spectra)")

    structures = eligible[["inchikey14", "normalized_smiles", "molecular_formula"]].drop_duplicates(
        subset=["inchikey14", "normalized_smiles"]
    )
    formula_map = {
        f: list(g[["inchikey14", "normalized_smiles"]].itertuples(index=False, name=None))
        for f, g in structures.groupby("molecular_formula")
    }

    if len(pool) > args.n_spectra:
        pool = pool.sample(args.n_spectra, random_state=args.seed)
    print(f"collecting matches for {len(pool)} spectra\n")

    all_rows: list[pd.DataFrame] = []
    t0 = time.time()
    for i, spec in enumerate(pool.itertuples(index=False)):
        candidates = [(spec.inchikey14, spec.normalized_smiles, False)]
        decoy_pool = [(k, s) for k, s in formula_map.get(spec.molecular_formula, []) if k != spec.inchikey14]
        if len(decoy_pool) > args.n_decoys:
            idx = rng.choice(len(decoy_pool), args.n_decoys, replace=False)
            decoy_pool = [decoy_pool[j] for j in idx]
        candidates += [(k, s, True) for k, s in decoy_pool]

        for key, smiles, is_decoy in candidates:
            try:
                frag = make_fragmentor(smiles, spec.adduct, args.tolerance_ppm, args.config)
                matches = frag.match_peaks(spec.ms2_mzs, spec.ms2_normalized_intensities)
            except Exception:
                continue
            if matches.empty:
                continue
            matches = matches.drop(columns=["fragment_smiles", "theo_mz"])
            matches["spectrum_id"] = spec.inchikey14
            matches["candidate_key"] = key
            matches["is_decoy"] = is_decoy
            matches["precursor_mz"] = spec.precursor_mz
            matches["adduct"] = spec.adduct
            matches["ionization_mode"] = spec.ionization_mode
            all_rows.append(matches)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pool)} spectra done ({time.time() - t0:.0f}s)", flush=True)

    dataset = pd.concat(all_rows, ignore_index=True)
    out = DATA_DIR / args.out
    dataset.to_parquet(out, index=False)
    print(f"\nsaved {len(dataset):,} match rows "
          f"({(dataset['is_decoy'] == False).sum():,} true / {(dataset['is_decoy'] == True).sum():,} decoy) to {out}")
    print(f"spectra represented: {dataset['spectrum_id'].nunique()}")


if __name__ == "__main__":
    main()

