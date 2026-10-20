"""Structure-level holdout simulation of the real competition task.

Class-2 simulation: sample structures, remove ALL their spectra from the
spectral library (their reference spectra are unknown), keep them in the
candidate pool (they stand in for 'external database' candidates), then
retrieve + rank candidates for each pseudo-molecule and score MRR@25.

Usage:
    uv run python scripts/simulate_holdout.py --n-molecules 200
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enveda_casmi26.pipeline import (
    CONFIGS,
    CandidatePool,
    ChannelEngine,
    Molecule,
    MoleculeSpectrum,
    SpectralLibrary,
    neutral_mass,
    rank_candidates,
    rank_gated,
    reciprocal_rank,
)
from enveda_casmi26.scoring import CandidateScorer

DATA_DIR = PROJECT_ROOT / "data"
BENCH_DIR = PROJECT_ROOT / "benchmark"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-molecules", type=int, default=200)
    parser.add_argument("--min-spectra", type=int, default=2)
    parser.add_argument("--max-spectra", type=int, default=6)
    parser.add_argument("--min-peaks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ppm", type=float, default=8.5)
    parser.add_argument("--ppm-fallback", type=float, default=30.0)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--mass-noise-ppm", type=float, default=3.0,
                        help="per-molecule query mass calibration error to simulate")
    parser.add_argument("--class1-frac", type=float, default=0.15,
                        help="fraction of holdout molecules whose other spectra stay in the library")
    parser.add_argument("--model", default="models/scorer.joblib")
    parser.add_argument("--configs", default="mass,sim,eif,learned",
                        help="comma-separated configs (see pipeline.CONFIGS)")
    parser.add_argument("--gated", action="store_true",
                        help="also evaluate sim-threshold gating (sim if >=0.9 else learned)")
    parser.add_argument("--save-scores", default=None,
                        help="path (relative to benchmark/) for per-candidate channel scores")
    parser.add_argument("--out", default="holdout_results.csv")
    args = parser.parse_args()

    print("Loading cleaned training data ...")
    tc = pd.read_parquet(
        DATA_DIR / "train_clean.parquet",
        columns=["inchikey14", "normalized_smiles", "adduct", "ionization_mode",
                 "precursor_mz", "num_peaks_clean", "drop_reason",
                 "ms2_mzs", "ms2_normalized_intensities"],
    )
    tc = tc[tc["inchikey14"].notna()]

    # structure table: neutral mass as median over its spectra
    tc["_nm"] = [neutral_mass(mz, ad) for mz, ad in zip(tc["precursor_mz"], tc["adduct"])]
    structs = (
        tc.groupby("inchikey14")
        .agg(normalized_smiles=("normalized_smiles", "first"),
             neutral_mass=("_nm", "median"),
             n_spectra=("inchikey14", "size"))
        .reset_index()
    )
    clean_counts = tc[tc["drop_reason"] == ""].groupby("inchikey14").size()
    structs = structs[structs["inchikey14"].isin(
        clean_counts[clean_counts >= args.min_spectra].index)]
    structs = structs[structs["n_spectra"].between(args.min_spectra, 16)]
    print(f"pool: {len(structs):,} structures")

    rng = np.random.default_rng(args.seed)
    holdout = structs.sample(min(args.n_molecules, len(structs)), random_state=args.seed)
    holdout_keys = set(holdout["inchikey14"])
    n_class1 = int(round(args.class1_frac * len(holdout_keys)))
    class1_keys = set(holdout["inchikey14"].sample(n_class1, random_state=args.seed))
    class2_keys = holdout_keys - class1_keys
    print(f"holdout molecules: {len(holdout_keys)} "
          f"({len(class2_keys)} class-2, {len(class1_keys)} class-1)")

    pool = CandidatePool(structs)
    library = SpectralLibrary(tc, exclude_keys=class2_keys)
    print(f"library: {len(library):,} structures (class-2 holdout excluded)")

    scorer = None
    if "learned" in args.configs or any("learned" in c for c in args.configs.split(",")):
        scorer = CandidateScorer.load(PROJECT_ROOT / args.model)
        print(f"loaded scorer from {args.model}")

    engine = ChannelEngine(pool, library, scorer=scorer)
    configs = {name: CONFIGS[name] for name in args.configs.split(",")}

    rows = []
    score_rows = []
    t0 = time.time()
    for i, key in enumerate(sorted(holdout_keys)):
        sub = tc[(tc["inchikey14"] == key) & (tc["drop_reason"] == "")].nlargest(
            args.max_spectra, "num_peaks_clean")
        spectra = [MoleculeSpectrum(np.asarray(r.ms2_mzs, dtype=float),
                                    np.asarray(r.ms2_normalized_intensities, dtype=float),
                                    r.adduct, r.precursor_mz, r.ionization_mode)
                   for r in sub.itertuples(index=False)]
        mol = Molecule(f"sim_{key}", spectra,
                       mass_noise_ppm=float(rng.normal(0, args.mass_noise_ppm)),
                       true_key=key)
        cand = pool.candidates_for(mol, args.ppm, args.ppm_fallback, args.max_candidates)
        rec = {"molecule_id": key, "n_spectra": len(spectra), "n_candidates": len(cand),
               "is_class1": key in class1_keys,
               "used_fallback": len(pool.window(mol.neutral_mass, args.ppm)) == 0}
        cand_keys = pool.keys[cand]
        true_in = key in cand_keys
        rec["true_in_window"] = true_in
        if true_in and len(cand) > 1:
            scores = engine.all_channels(mol, cand)
            if args.save_scores:
                for j in range(len(cand)):
                    score_rows.append({
                        "molecule_id": key, "inchikey14": pool.keys[cand[j]],
                        "is_class1": key in class1_keys,
                        "is_true": pool.keys[cand[j]] == key,
                        "mass_delta_ppm": (pool.masses[cand[j]] - mol.neutral_mass)
                                          / mol.neutral_mass * 1e6,
                        "sim": float(scores["sim"][j]),
                        "eif": float(scores["eif"][j]),
                        "learned": float(scores["learned"][j]),
                    })
            for name, config in configs.items():
                ranked_keys = pool.keys[rank_candidates(scores, config, cand)]
                rec[f"rr_{name}"] = reciprocal_rank(ranked_keys, key)
                rec[f"rank_{name}"] = int(np.flatnonzero(ranked_keys == key)[0]) + 1
            if args.gated:
                ranked_keys = pool.keys[rank_gated(scores, cand)]
                rec["rr_gated"] = reciprocal_rank(ranked_keys, key)
                rec["rank_gated"] = int(np.flatnonzero(ranked_keys == key)[0]) + 1
        elif true_in:
            for name in configs:
                rec[f"rr_{name}"] = 1.0
                rec[f"rank_{name}"] = 1
            if args.gated:
                rec["rr_gated"] = 1.0
                rec["rank_gated"] = 1
        else:
            for name in configs:
                rec[f"rr_{name}"] = 0.0
                rec[f"rank_{name}"] = 999
            if args.gated:
                rec["rr_gated"] = 0.0
                rec["rank_gated"] = 999
        rows.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(holdout_keys)} ({time.time() - t0:.0f}s)")

    out = pd.DataFrame(rows)
    BENCH_DIR.mkdir(exist_ok=True)
    out.to_csv(BENCH_DIR / args.out, index=False)
    if args.save_scores:
        pd.DataFrame(score_rows).to_parquet(BENCH_DIR / args.save_scores)
        print(f"channel scores for {len(score_rows):,} candidates saved to "
              f"{BENCH_DIR / args.save_scores}")

    print(f"\n=== holdout simulation ({len(out)} molecules, "
          f"{args.ppm}/{args.ppm_fallback} ppm windows) ===")
    print(f"true structure in candidate window: {out['true_in_window'].mean():.1%}")
    print(f"candidates per molecule: median {out['n_candidates'].median():.0f}, "
          f"p90 {out['n_candidates'].quantile(0.9):.0f}")
    print(f"\n{'config':<18} {'MRR@25':>8} {'top-1':>8} {'top-3':>8} {'top-25':>8}")
    names = list(configs) + (["gated"] if args.gated else [])
    for name in names:
        rr = out[f"rr_{name}"]
        rank = out[f"rank_{name}"]
        print(f"{name:<18} {rr.mean():>8.3f} {(rank == 1).mean():>8.1%} "
              f"{(rank <= 3).mean():>8.1%} {(rank <= 25).mean():>8.1%}")
    c2 = out[~out["is_class1"]]
    if len(c2) and out["is_class1"].any():
        print(f"\nclass-2 only ({len(c2)} molecules):")
        for name in names:
            print(f"{name:<18} {c2[f'rr_{name}'].mean():>8.3f} "
                  f"{(c2[f'rank_{name}'] == 1).mean():>8.1%}")


if __name__ == "__main__":
    main()
