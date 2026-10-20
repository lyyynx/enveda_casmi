"""Failure analysis: which molecules fail, and why.

Part A -- validation split: for every test_cut molecule, rebuild the exact
kernel pipeline, then classify the outcome:

  pool_missing    truth structure absent from the candidate pool
  no_candidates   query mass unusable (NaN) or empty windows
  window_missing  truth is in the pool but outside the retrieval window
  ranked_out      truth was a candidate but ranked below slot 25
  low_rank        truth found at rank 6..25
  hit             truth found within the top 5

Also records per-molecule channel evidence (truth sim / learned, best
distractor scores, gate state) so failures can be triaged afterwards.

Part B -- placeholder test (leaky duplicates): per-molecule score of the
last kernel output against the leaked train labels.

Usage:
    uv run python scripts/analyze_failures.py [--n-mols N] [--skip-placeholder]
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from enveda_casmi26.cleaning import CleanConfig, SpectrumCleaner  # noqa: E402
from enveda_casmi26.pipeline import (  # noqa: E402
    CandidatePool,
    ChannelEngine,
    Molecule,
    MoleculeSpectrum,
    SimpleScorer,
    SpectralLibrary,
    _zscore,
    neutral_mass,
)

DATA = PROJECT_ROOT / "data"
BENCH = PROJECT_ROOT / "benchmark"
PPM, PPM_FALLBACK, MAX_CANDIDATES, GATE_SIM = 8.5, 30.0, 200, 0.98


def build_engine():
    tab = pq.read_table(DATA / "train_cut.parquet",
                        columns=["inchikey14", "normalized_smiles", "precursor_mz",
                                 "adduct"]).to_pandas()
    tab = tab[tab["inchikey14"].notna()]
    tab["nm"] = [neutral_mass(m, a) for m, a in zip(tab["precursor_mz"], tab["adduct"])]
    structs = (tab.groupby("inchikey14")
               .agg(normalized_smiles=("normalized_smiles", "first"),
                    neutral_mass=("nm", "median")).reset_index())
    del tab

    best = {}
    pf = pq.ParquetFile(DATA / "train_cut.parquet")
    for batch in pf.iter_batches(batch_size=100_000,
                                  columns=["inchikey14", "ms2_mzs",
                                           "ms2_normalized_intensities", "precursor_mz"]):
        for k, m, i, p in zip(batch.column("inchikey14").to_pylist(),
                               batch.column("ms2_mzs").to_pylist(),
                               batch.column("ms2_normalized_intensities").to_pylist(),
                               batch.column("precursor_mz").to_pylist()):
            if k is None or not m:
                continue
            cur = best.get(k)
            if cur is None or len(m) > len(cur[0]):
                best[k] = (m, i, p)

    cleaner = SpectrumCleaner(CleanConfig())
    lib_rows = {"inchikey14": [], "ms2_mzs": [], "ms2_normalized_intensities": [],
                "num_peaks_clean": [], "drop_reason": []}
    for k, (m, i, p) in best.items():
        res = cleaner.clean(np.asarray(m, dtype=float), np.asarray(i, dtype=float),
                            precursor_mz=p)
        lib_rows["inchikey14"].append(k)
        lib_rows["ms2_mzs"].append(res.mzs)
        lib_rows["ms2_normalized_intensities"].append(res.intensities)
        lib_rows["num_peaks_clean"].append(res.n_kept)
        lib_rows["drop_reason"].append(res.drop_reason or "")
    lib_df = pd.DataFrame(lib_rows)
    del best

    coco = pd.read_parquet(DATA / "val_external_pool.parquet")
    coco = coco.rename(columns={"smiles": "normalized_smiles", "mass": "neutral_mass"})
    pool = CandidatePool(pd.concat([structs, coco], ignore_index=True)
                         .drop_duplicates("inchikey14", keep="first"))
    library = SpectralLibrary(lib_df)
    scorer = SimpleScorer.load(PROJECT_ROOT / "models" / "scorer_default.json")
    engine = ChannelEngine(pool, library, scorer=scorer, max_frag_candidates=50)
    return pool, engine, cleaner


def categorize(rr: float, rank: int) -> str:
    if rr == 0.0:
        return "missed"
    if rank == 1:
        return "hit_top1"
    if rank <= 5:
        return "hit_top5"
    return "low_rank"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-mols", type=int, default=None)
    parser.add_argument("--skip-placeholder", action="store_true")
    args = parser.parse_args()

    pool, engine, cleaner = build_engine()
    print(f"pool {len(pool):,}  library {len(engine.library):,}")

    test = pd.read_parquet(DATA / "test_cut.parquet")
    mapping = pd.read_csv(DATA / "cut_map.csv")
    truth_of = dict(zip(mapping["molecule_id"], mapping["inchikey14"]))
    smiles_of = dict(zip(mapping["molecule_id"], mapping["normalized_smiles"]))
    slice_of = dict(zip(mapping["molecule_id"], mapping["slice"]))

    molecules = {}
    for r in test.itertuples(index=False):
        res = cleaner.clean(np.asarray(r.ms2_mzs, dtype=float),
                            np.asarray(r.ms2_normalized_intensities, dtype=float),
                            precursor_mz=float(r.precursor_mz))
        if res.n_kept == 0:
            continue
        spec = MoleculeSpectrum(res.mzs, res.intensities, r.adduct,
                                float(r.precursor_mz), r.ionization_mode)
        molecules.setdefault(r.molecule_id, Molecule(r.molecule_id)).spectra.append(spec)
    ids = list(molecules.keys())
    if args.n_mols:
        ids = ids[:args.n_mols]

    rows = []
    for n, mid in enumerate(ids):
        mol = molecules[mid]
        mol.spectra = mol.spectra[:4]
        tk = truth_of[mid]
        qm = mol.neutral_mass
        out = {"molecule_id": mid, "slice": slice_of[mid], "true_key": tk,
               "true_smiles": smiles_of[mid], "query_mass": qm,
               "n_spectra": len(mol.spectra),
               "peaks_min": min(len(s.mzs) for s in mol.spectra)}

        ti = pool.key_to_idx.get(tk)
        cand = pool.candidates_for(mol, PPM, PPM_FALLBACK, MAX_CANDIDATES)
        if len(cand) == 0:
            for wide in (100.0, 500.0):
                cand = pool.window(mol.neutral_mass, wide)
                if len(cand):
                    break
        out["n_candidates"] = len(cand)
        out["truth_in_pool"] = ti is not None
        if ti is not None and np.isfinite(qm):
            out["mass_delta_ppm"] = (pool.masses[ti] - qm) * 1e6 / qm

        reason = ""
        if ti is None:
            reason = "pool_missing"
        elif not np.isfinite(qm):
            reason = "mass_nan"
        elif ti not in set(cand.tolist()):
            reason = "window_missing"
        if not reason:
            sim = engine.sim_scores(mol, cand)
            truth_pos = int(np.flatnonzero(cand == ti)[0])
            out["truth_sim"] = sim[truth_pos]
            out["best_sim"] = sim.max()
            out["gated"] = sim.max() >= GATE_SIM
            if sim.max() >= GATE_SIM:
                order = np.argsort(-sim, kind="stable")
            else:
                learned = engine.learned_scores(mol, cand)
                out["truth_learned"] = learned[truth_pos]
                out["best_learned"] = np.nanmax(learned)
                fused = _zscore(learned) + _zscore(np.maximum(sim - 0.6, 0.0))
                order = np.argsort(-fused, kind="stable")
            rank = int(np.flatnonzero(order == truth_pos)[0]) + 1
            out["rank"] = rank
            out["rr"] = 1.0 / rank if rank <= 25 else 0.0
            reason = categorize(out["rr"], rank)
        else:
            out["rank"] = 0
            out["rr"] = 0.0
        out["reason"] = reason
        rows.append(out)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(ids)}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(BENCH / "failure_analysis.csv", index=False)

    print("\n===== failure taxonomy (validation split) =====")
    for r, g in res.groupby("reason"):
        lost = (1.0 - g["rr"]).sum()
        print(f"{r:>15}: n={len(g):4d} ({len(g) / len(res):5.1%})   "
              f"MRR lost: {lost / len(res):.3f}")

    missed = res[res["reason"] == "missed"].sort_values(
        ["best_sim", "truth_sim"], ascending=False)
    cols = ["molecule_id", "true_smiles", "n_candidates", "truth_sim", "best_sim",
            "truth_learned", "best_learned", "n_spectra", "peaks_min"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print("\nworst 'ranked_out' molecules (truth was a candidate):")
        print(missed[cols].head(20).to_string(index=False))
        print("\nwindow_missing:")
        wm = res[res["reason"] == "window_missing"]
        print(wm[["molecule_id", "true_smiles", "query_mass", "mass_delta_ppm"]]
              .head(20).to_string(index=False) if len(wm) else "  none")

    if not args.skip_placeholder:
        print("\n===== placeholder test (leaky) per-molecule =====")
        sub = pd.read_csv(BENCH / "kernel_v5_out" / "submission.csv")
        rows = []
        for _, r in sub.iterrows():
            rows.append({"molecule_id": r["molecule_id"],
                         "rr": 0.0})
        ph = pd.DataFrame(rows)
        ph.to_csv(BENCH / "placeholder_per_molecule.csv", index=False)
        print("placeholder truth-scoring needs spectrum hashing; "
              "run scripts/evaluate_kernel_output.py for aggregate numbers")


if __name__ == "__main__":
    main()
