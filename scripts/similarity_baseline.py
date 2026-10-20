"""Spectral similarity baseline: library search against train structures.

Modes:
  test       search all test spectra against the train structure library,
             aggregate best hits per test molecule
  validate   hold out train spectra as pseudo-queries, measure retrieval
             top-1/top-3 of the true structure
  decoys     score the decoy-benchmark candidates (true + 9 same-formula
             decoys) by similarity for direct comparison with the fragmenter
             scorer

Usage:
    uv run python scripts/similarity_baseline.py test
    uv run python scripts/similarity_baseline.py validate --n-queries 300
    uv run python scripts/similarity_baseline.py decoys
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BENCH_DIR = PROJECT_ROOT / "benchmark"

BIN_WIDTH = 0.05
N_BINS = 40_000
SCREEN_TOP_K = 30
REFINE_TOL = 0.02


def refined_cosine(mzs_a, ints_a, mzs_b, ints_b, tol: float = REFINE_TOL) -> float:
    na = float(np.linalg.norm(ints_a))
    nb = float(np.linalg.norm(ints_b))
    if na == 0 or nb == 0:
        return 0.0
    i = j = 0
    dot = 0.0
    while i < len(mzs_a) and j < len(mzs_b):
        d = mzs_a[i] - mzs_b[j]
        if abs(d) <= tol:
            dot += ints_a[i] * ints_b[j]
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return dot / (na * nb)


def bin_spectrum(mzs, ints):
    """Sparse-binned, L2-normalized vector as (indices, data) arrays."""
    mzs = np.asarray(mzs, dtype=float)
    ints = np.asarray(ints, dtype=np.float32)
    if len(mzs) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    bins = np.clip((mzs / BIN_WIDTH).astype(np.int64), 0, N_BINS - 1)
    order = np.argsort(bins, kind="stable")
    bins_s, ints_s = bins[order], ints[order]
    starts_mask = np.empty(len(bins_s), dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = bins_s[1:] != bins_s[:-1]
    starts = np.flatnonzero(starts_mask)
    summed = np.add.reduceat(ints_s, starts).astype(np.float32)
    norm = float(np.sqrt(np.sum(summed ** 2)))
    if norm == 0:
        return bins_s[starts], summed * 0.0
    return bins_s[starts], summed / norm


class Library:
    """One representative (and one backup) spectrum per train structure."""

    def __init__(self) -> None:
        tc = pd.read_parquet(
            DATA_DIR / "train_clean.parquet",
            columns=["inchikey14", "normalized_smiles", "adduct", "ionization_mode",
                     "precursor_mz", "num_peaks_clean", "drop_reason",
                     "ms2_mzs", "ms2_normalized_intensities"],
        )
        tc["_ok"] = (tc["drop_reason"] == "").astype(int)
        tc = tc.sort_values(["inchikey14", "_ok", "num_peaks_clean"],
                            ascending=[True, False, False])
        top2 = tc.groupby("inchikey14", sort=False).head(2)

        self.reps = top2.drop_duplicates("inchikey14", keep="first").reset_index(drop=True)
        self.backup = top2[top2.duplicated("inchikey14", keep="first")].reset_index(drop=True)
        backup_keys = self.backup["inchikey14"].to_numpy()
        self.backup_by_key = {k: i for i, k in enumerate(backup_keys)}
        self.keys = self.reps["inchikey14"].to_numpy()
        self.smiles = self.reps["normalized_smiles"].to_numpy()
        self.polarity = self.reps["ionization_mode"].to_numpy()
        self.n = len(self.reps)

        indptr = np.zeros(self.n + 1, dtype=np.int64)
        indices_parts, data_parts = [], []
        for i, (mzs, ints) in enumerate(zip(self.reps["ms2_mzs"],
                                            self.reps["ms2_normalized_intensities"])):
            idx, val = bin_spectrum(mzs, ints)
            indices_parts.append(idx)
            data_parts.append(val)
            indptr[i + 1] = indptr[i] + len(idx)
        indices = np.concatenate(indices_parts)
        data = np.concatenate(data_parts)
        self.mat = csr_matrix((data, indices, indptr), shape=(self.n, N_BINS),
                               dtype=np.float32)

    def arrays(self, structure_idx: int, exclude_arrays=None):
        """mzs/intensities of the representative (or backup if excluded)."""
        row = self.reps.iloc[structure_idx]
        mzs = np.asarray(row["ms2_mzs"], dtype=float)
        if exclude_arrays is not None and np.array_equal(mzs, exclude_arrays):
            j = self.backup_by_key.get(row["inchikey14"])
            if j is not None:
                alt = self.backup.iloc[j]
                if not np.array_equal(np.asarray(alt["ms2_mzs"], dtype=float), exclude_arrays):
                    return (np.asarray(alt["ms2_mzs"], dtype=float),
                            np.asarray(alt["ms2_normalized_intensities"], dtype=float))
            return None
        return mzs, np.asarray(row["ms2_normalized_intensities"], dtype=float)

    def screen(self, mzs, ints, polarity: str, top_k: int = SCREEN_TOP_K) -> np.ndarray:
        bins, vals = bin_spectrum(mzs, ints)
        if len(bins) == 0:
            return np.array([], dtype=np.int64)
        q = np.zeros(N_BINS, dtype=np.float32)
        q[bins] = vals
        scores = np.asarray(self.mat @ q).ravel()
        scores[~(self.polarity == polarity)] = 0.0
        k = min(top_k, self.n)
        return np.argpartition(-scores, k - 1)[:k]

    def search(self, mzs, ints, polarity: str, top_k: int = 10, exclude_arrays=None):
        cand = self.screen(mzs, ints, polarity, top_k=max(SCREEN_TOP_K, top_k))
        q_mzs = np.asarray(mzs, dtype=float)
        q_ints = np.asarray(ints, dtype=float)
        results = []
        for idx in cand:
            arrays = self.arrays(int(idx), exclude_arrays=exclude_arrays)
            if arrays is None:
                continue
            results.append((int(idx),
                            refined_cosine(q_mzs, q_ints, arrays[0], arrays[1])))
        results.sort(key=lambda t: -t[1])
        return results[:top_k]


def eligible_spectra(min_peaks: int):
    df = pd.read_parquet(
        DATA_DIR / "train_clean.parquet",
        columns=["inchikey14", "normalized_smiles", "molecular_formula", "adduct", "ionization_mode",
                 "num_peaks_clean", "drop_reason", "ms2_mzs", "ms2_normalized_intensities"],
    )
    return df[(df["drop_reason"] == "") & (df["num_peaks_clean"] >= min_peaks) & df["inchikey14"].notna()]


def run_test(lib: Library, args) -> None:
    test = pd.read_parquet(DATA_DIR / "test_clean.parquet")
    key_to_idx = {k: i for i, k in enumerate(lib.keys)}
    rows = []
    for mol, group in test.groupby("molecule_id"):
        best_per_structure: dict[str, float] = {}
        for spec in group.itertuples(index=False):
            hits = lib.search(spec.ms2_mzs, spec.ms2_normalized_intensities, spec.ionization_mode)
            for idx, score in hits:
                key = lib.keys[idx]
                best_per_structure[key] = max(best_per_structure.get(key, 0.0), score)
        ranked = sorted(best_per_structure.items(), key=lambda t: -t[1])[:10]
        for rank, (key, score) in enumerate(ranked, start=1):
            rep_i = key_to_idx[key]
            rows.append({
                "molecule_id": mol, "rank": rank, "inchikey14": key,
                "normalized_smiles": lib.smiles[rep_i],
                "adduct": lib.reps.iloc[rep_i]["adduct"],
                "best_cosine": round(score, 4), "n_query_spectra": len(group),
            })
    out = pd.DataFrame(rows)
    BENCH_DIR.mkdir(exist_ok=True)
    out.to_csv(BENCH_DIR / "test_similarity_hits.csv", index=False)

    per_mol = out.groupby("molecule_id")["best_cosine"].max()
    print(f"\ntest molecules: {per_mol.index.nunique()}")
    for thr in (0.95, 0.9, 0.8, 0.6, 0.4):
        print(f"  best cosine >= {thr}: {(per_mol >= thr).sum()} molecules ({(per_mol >= thr).mean():.1%})")
    print(f"  median best cosine: {per_mol.median():.3f}")
    print("\nexample top hits (highest similarity):")
    print(out.sort_values("best_cosine", ascending=False).head(10).to_string(index=False))
    print(f"\nper-molecule hits saved to {BENCH_DIR / 'test_similarity_hits.csv'}")


def run_validate(lib: Library, args) -> None:
    tc = eligible_spectra(args.min_peaks)
    queries = tc.sample(min(args.n_queries, len(tc)), random_state=11)

    top1 = top3 = top10 = trivial = 0
    ranks = []
    t0 = time.time()
    for i, spec in enumerate(queries.itertuples(index=False)):
        q_mzs = np.asarray(spec.ms2_mzs, dtype=float)
        q_ints = np.asarray(spec.ms2_normalized_intensities, dtype=float)
        hits = lib.search(q_mzs, q_ints, spec.ionization_mode, top_k=50, exclude_arrays=q_mzs)
        rank = next((r for r, (idx, _) in enumerate(hits, start=1)
                     if lib.keys[idx] == spec.inchikey14), 999)
        if rank == 999:
            fallback = lib.search(q_mzs, q_ints, spec.ionization_mode, top_k=50)
            if any(lib.keys[idx] == spec.inchikey14 for idx, _ in fallback):
                trivial += 1
        ranks.append(rank)
        top1 += rank == 1
        top3 += rank <= 3
        top10 += rank <= 10
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(queries)} ({time.time() - t0:.0f}s)")
    ranks = np.array(ranks)
    print(f"\nlibrary search on {len(queries)} held-out train spectra:")
    print(f"  top-1: {top1 / len(queries):.1%}   top-3: {top3 / len(queries):.1%}   top-10: {top10 / len(queries):.1%}")
    print(f"  median rank of true structure: {np.median(ranks):.0f}")
    print(f"  queries only findable via exact duplicate spectrum: {trivial}")


def run_decoys(lib: Library, args) -> None:
    from enveda_casmi26.fragmentation import ADDUCTS, ADDUCT_ALIASES

    supported_adducts = set(ADDUCTS) | set(ADDUCT_ALIASES)
    df = pd.read_parquet(
        DATA_DIR / "train_clean.parquet",
        columns=["inchikey14", "normalized_smiles", "molecular_formula", "adduct",
                 "ionization_mode", "num_peaks_clean", "drop_reason",
                 "ms2_mzs", "ms2_normalized_intensities"],
    )
    df = df[
        (df["drop_reason"] == "")
        & (df["num_peaks_clean"] >= 10)
        & (df["adduct"].isin(supported_adducts))
        & df["inchikey14"].notna()
    ]
    df = df.drop_duplicates("inchikey14", keep="first")
    structures = df[["inchikey14", "normalized_smiles", "molecular_formula"]].drop_duplicates(
        subset=["inchikey14", "normalized_smiles"])
    formula_map = {f: list(g[["inchikey14", "normalized_smiles"]].itertuples(index=False, name=None))
                   for f, g in structures.groupby("molecular_formula")}

    queries = df.sample(150, random_state=42)
    rng = np.random.default_rng(42)
    key_to_idx = {k: i for i, k in enumerate(lib.keys)}

    top1 = top3 = n_eval = skipped = 0
    margins = []
    rows = []
    for spec in queries.itertuples(index=False):
        pool = [(k, s) for k, s in formula_map.get(spec.molecular_formula, []) if k != spec.inchikey14]
        if not pool:
            skipped += 1
            continue
        decoys = pool if len(pool) <= 9 else [pool[j] for j in rng.choice(len(pool), 9, replace=False)]
        q_mzs = np.asarray(spec.ms2_mzs, dtype=float)
        q_ints = np.asarray(spec.ms2_normalized_intensities, dtype=float)
        scores = []
        for key, _ in [(spec.inchikey14, spec.normalized_smiles)] + decoys:
            arrays = lib.arrays(key_to_idx[key], exclude_arrays=q_mzs)
            if arrays is None:
                scores.append(0.0)
                continue
            scores.append(refined_cosine(q_mzs, q_ints, arrays[0], arrays[1]))
        true_score = scores[0]
        decoy_scores = scores[1:]
        n_eval += 1
        rank = 1 + sum(s >= true_score for s in decoy_scores)
        top1 += rank == 1
        top3 += rank <= 3
        margins.append(true_score - max(decoy_scores))
        rows.append({"inchikey14": spec.inchikey14, "n_candidates": 1 + len(decoys),
                     "sim_true_score": true_score,
                     "sim_best_decoy": max(decoy_scores), "sim_rank": rank})
    print(f"\ndecoy benchmark via similarity ({n_eval} spectra, skipped {skipped}):")
    print(f"  top-1: {top1 / n_eval:.1%}   top-3: {top3 / n_eval:.1%}")
    print(f"  mean margin (true - best decoy): {np.mean(margins):+.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(BENCH_DIR / "decoy_similarity_results.csv", index=False)
    print(f"per-spectrum results saved to {BENCH_DIR / 'decoy_similarity_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["test", "validate", "decoys"])
    parser.add_argument("--n-queries", type=int, default=300)
    parser.add_argument("--min-peaks", type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()
    print("Building structure library ...")
    lib = Library()
    print(f"library: {lib.n:,} structures ({time.time() - t0:.0f}s)")

    {"test": run_test, "validate": run_validate, "decoys": run_decoys}[args.mode](lib, args)


if __name__ == "__main__":
    main()
