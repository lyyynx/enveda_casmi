"""Decoy benchmark: can fragmenter-based scoring rank the true structure above
same-formula decoy structures?

For each sampled training spectrum, candidate structures are the true structure
plus N decoys drawn from other training structures with the same molecular
formula. Each candidate is scored by explained intensity fraction of the
observed (cleaned) spectrum, and we report the rank of the true structure.

Usage:
    uv run python scripts/benchmark_decoys.py --label new
    uv run python scripts/benchmark_decoys.py --label legacy --legacy
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from enveda_casmi26.fragmentation import ADDUCTS, ADDUCT_ALIASES, LOSS_DEFS, Fragmentor
from enveda_casmi26.scoring import CandidateScorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BENCH_DIR = PROJECT_ROOT / "benchmark"
MODEL_PATH = PROJECT_ROOT / "models" / "scorer.joblib"
SUPPORTED_ADDUCTS = set(ADDUCTS) | set(ADDUCT_ALIASES)


def make_fragmentor(smiles: str, adduct: str, args: argparse.Namespace) -> Fragmentor:
    if args.legacy:
        return Fragmentor(
            smiles,
            adduct=adduct,
            tolerance_ppm=args.tolerance_ppm,
            h_shifts=(-1, 0, 1),
            neutral_losses=("H2O", "NH3", "CO", "CO2", "CH3OH"),
            min_mz=50.0,
            fg_gating=False,
            max_serial_losses=1,
            enforce_even_electron=False,
        )
    kwargs = {
        "include_precursor_losses": args.with_precursor_losses,
        "min_mz": args.min_mz,
    }
    if args.serial_losses is not None:
        kwargs["max_serial_losses"] = args.serial_losses
    if args.losses:
        kwargs["neutral_losses"] = tuple(args.losses.split(","))
    if args.h_shifts:
        kwargs["h_shifts"] = tuple(int(x) for x in args.h_shifts.split(","))
    if args.no_parity:
        kwargs["enforce_even_electron"] = False
    if args.no_gating:
        kwargs["fg_gating"] = False
    return Fragmentor(smiles, adduct=adduct, tolerance_ppm=args.tolerance_ppm, **kwargs)


def score_candidate(smiles: str, spec, args: argparse.Namespace, scorer=None) -> float:
    try:
        frag = make_fragmentor(smiles, spec.adduct, args)
        return frag.explained_intensity_fraction(spec.ms2_mzs, spec.ms2_normalized_intensities)
    except Exception:
        return np.nan


def learned_scores(candidates: list[tuple[str, str]], spec, args: argparse.Namespace, scorer) -> list[float]:
    """Score all candidates of one spectrum together with the learned model."""
    rows = []
    for _, smiles in candidates:
        try:
            frag = Fragmentor(
                smiles,
                adduct=spec.adduct,
                tolerance_ppm=args.tolerance_ppm,
                include_precursor_losses=False,
                max_serial_losses=2 if args.discovery else None,
                neutral_losses=tuple(LOSS_DEFS) if args.discovery else None,
                h_shifts=(-2, -1, 0, 1, 2) if args.discovery else None,
            )
            from enveda_casmi26.scoring import candidate_row
            rows.append(candidate_row(frag, spec.ms2_mzs, spec.ms2_normalized_intensities, spec.ionization_mode))
        except Exception:
            rows.append(None)
    valid = [i for i, r in enumerate(rows) if r is not None]
    if not valid:
        return [np.nan] * len(rows)
    probs = scorer.score_set([rows[i] for i in valid])
    return [probs[valid.index(i)] if i in valid else np.nan for i in range(len(rows))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-spectra", type=int, default=150)
    parser.add_argument("--n-decoys", type=int, default=9)
    parser.add_argument("--min-peaks", type=int, default=10)
    parser.add_argument("--tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy", action="store_true", help="use pre-Tier1 fragmenter settings")
    parser.add_argument("--with-precursor-losses", action="store_true",
                        help="include formula-only precursor losses in scoring (hurts discrimination)")
    parser.add_argument("--no-parity", action="store_true")
    parser.add_argument("--no-gating", action="store_true")
    parser.add_argument("--serial-losses", type=int, default=None)
    parser.add_argument("--min-mz", type=float, default=25.0)
    parser.add_argument("--losses", default=None, help="comma-separated loss names")
    parser.add_argument("--h-shifts", default=None, help="comma-separated h shifts, e.g. -2,-1,0,1,2")
    parser.add_argument("--label", default="new")
    parser.add_argument("--learned", action="store_true", help="score with the trained likelihood model")
    parser.add_argument("--model-path", default=None, help="scorer model to use with --learned")
    parser.add_argument("--discovery", action="store_true",
                        help="use the permissive discovery fragmenter config (needs a matching model)")
    args = parser.parse_args()

    scorer = None
    if args.learned:
        model_path = Path(args.model_path) if args.model_path else MODEL_PATH
        scorer = CandidateScorer.load(model_path)
        print(f"loaded learned scorer from {model_path}")

    rng = np.random.default_rng(args.seed)

    print("Loading cleaned training data ...")
    df = pd.read_parquet(DATA_DIR / "train_clean.parquet")
    df = df[
        (df["drop_reason"] == "")
        & (df["num_peaks_clean"] >= args.min_peaks)
        & (df["adduct"].isin(SUPPORTED_ADDUCTS))
        & df["inchikey14"].notna()
    ]
    df = df.drop_duplicates(subset="inchikey14", keep="first")
    print(f"eligible spectra: {len(df):,}")

    structures = (
        df[["inchikey14", "normalized_smiles", "molecular_formula"]]
        .drop_duplicates(subset=["inchikey14", "normalized_smiles"])
    )
    formula_map = {f: list(g[["inchikey14", "normalized_smiles"]].itertuples(index=False, name=None))
                   for f, g in structures.groupby("molecular_formula")}

    if len(df) > args.n_spectra:
        df = df.sample(args.n_spectra, random_state=args.seed)
    print(f"benchmarking {len(df)} spectra with up to {args.n_decoys} decoys each\n")

    rows = []
    skipped = 0
    t0 = time.time()
    for i, spec in enumerate(df.itertuples(index=False)):
        pool = [(k, s) for k, s in formula_map.get(spec.molecular_formula, []) if k != spec.inchikey14]
        if not pool:
            skipped += 1
            continue
        decoys = pool if len(pool) <= args.n_decoys else rng.choice(len(pool), args.n_decoys, replace=False).tolist()
        decoys = [pool[j] if isinstance(j, (int, np.integer)) else j for j in decoys]

        if args.learned:
            all_candidates = [(spec.inchikey14, spec.normalized_smiles)] + decoys
            all_scores = learned_scores(all_candidates, spec, args, scorer)
            true_score = all_scores[0]
            decoy_scores = all_scores[1:]
        else:
            true_score = score_candidate(spec.normalized_smiles, spec, args)
            decoy_scores = [score_candidate(s, spec, args) for _, s in decoys]

        valid = [s for s in decoy_scores if np.isfinite(s)]
        if not np.isfinite(true_score):
            rank = 1 + len(decoy_scores)
        else:
            rank = 1 + sum(s >= true_score for s in valid)
        rows.append({
            "inchikey14": spec.inchikey14,
            "formula": spec.molecular_formula,
            "adduct": spec.adduct,
            "n_peaks": len(spec.ms2_mzs),
            "n_candidates": 1 + len(valid),
            "true_score": true_score,
            "best_decoy_score": max(valid) if valid else np.nan,
            "margin": true_score - (max(valid) if valid else np.nan),
            "rank": rank,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(df)} spectra done ({time.time() - t0:.0f}s)")

    results = pd.DataFrame(rows)
    if results.empty:
        print("No spectra with same-formula decoys found.")
        return

    BENCH_DIR.mkdir(exist_ok=True)
    out = BENCH_DIR / f"decoy_results_{args.label}.csv"
    results.to_csv(out, index=False)

    top1 = (results["rank"] == 1).mean()
    top3 = (results["rank"] <= 3).mean()
    print(f"\n===== Benchmark: {args.label} =====")
    print(f"  evaluated spectra: {len(results)}  (skipped {skipped}: no same-formula decoys)")
    print(f"  top-1 accuracy:  {top1:.1%}")
    print(f"  top-3 accuracy:  {top3:.1%}")
    print(f"  mean rank of true structure: {results['rank'].mean():.2f}")
    print(f"  mean true score (explained intensity): {results['true_score'].mean():.1%}")
    print(f"  mean best-decoy score: {results['best_decoy_score'].mean():.1%}")
    print(f"  mean margin (true - best decoy): {results['margin'].mean():+.1%}")
    print(f"  results saved to {out}")


if __name__ == "__main__":
    main()
