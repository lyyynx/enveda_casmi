"""Mine systematically unexplained MS2 peaks in training spectra.

For each sampled structure, fragment the TRUE structure with the most
permissive theory (full loss dictionary, serial loss cascades, precursor
losses, +-2 H shifts) and record peaks with NO theoretical match. Peaks that
remain unexplained even under the permissive config indicate missing
fragmentation chemistry (new loss/rearrangement rules) or candidate
diagnostic ions (prior features).

Hygiene: enveda-180 and the validation holdout are excluded.

Usage:
    uv run python scripts/mine_unexplained.py [--n-structs 12000]
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from enveda_casmi26.fragmentation import (ADDUCTS, ADDUCT_ALIASES, LOSS_DEFS,
                                          Fragmentor, PROTON_MASS, MONOISOTOPIC_MASSES)
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-structs", type=int, default=12000)
    ap.add_argument("--tol-ppm", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--out", default="benchmark/unexplained_peaks.csv")
    args = ap.parse_args()

    print("loading cleaned training data ...", flush=True)
    tc = pd.read_parquet(PROJECT_ROOT / "data" / "train_clean.parquet",
                         columns=["inchikey14", "normalized_smiles", "ingest_lib",
                                  "precursor_mz", "adduct", "ms2_mzs",
                                  "ms2_normalized_intensities",
                                  "num_peaks_clean", "drop_reason"])
    tc = tc[(tc["drop_reason"] == "") & (tc["num_peaks_clean"] >= 10)
            & (tc["adduct"].isin(set(ADDUCTS) | set(ADDUCT_ALIASES)))
            & tc["inchikey14"].notna() & tc["normalized_smiles"].notna()]
    tc = tc[tc["ingest_lib"] != "enveda-180"]
    holdout = set(pd.read_parquet(PROJECT_ROOT / "data" / "test_cut.parquet",
                                  columns=["inchikey14"])["inchikey14"])
    tc = tc[~tc["inchikey14"].isin(holdout)]
    tc = tc.drop_duplicates(subset="inchikey14", keep="first")
    if len(tc) > args.n_structs:
        tc = tc.sample(args.n_structs, random_state=args.seed)
    print(f"mining over {len(tc):,} structures", flush=True)

    rows = []
    t0 = time.time()
    total_peaks = 0
    total_intensity = 0.0
    unexplained_intensity = 0.0
    for i, spec in enumerate(tc.itertuples(index=False)):
        try:
            frag = Fragmentor(spec.normalized_smiles, adduct=spec.adduct,
                              tolerance_ppm=args.tol_ppm,
                              include_precursor_losses=True,
                              max_serial_losses=2,
                              include_diagnostics=not args.no_diagnostics,
                              neutral_losses=tuple(LOSS_DEFS))
        except Exception:
            continue
        mzs = np.asarray(spec.ms2_mzs, dtype=float)
        ints = np.asarray(spec.ms2_normalized_intensities, dtype=float)
        isum = float(ints.sum())
        if isum <= 0:
            continue
        try:
            m = frag.match_peaks(mzs, ints, best_only=False)
        except Exception:
            continue
        explained = set(m["peak_index"].unique()) if not m.empty else set()
        total_peaks += len(mzs)
        total_intensity += isum
        for pi in range(len(mzs)):
            if pi in explained:
                continue
            unexplained_intensity += float(ints[pi])
            rows.append({"mz": float(mzs[pi]), "intensity": float(ints[pi]),
                         "share": float(ints[pi]) / isum, "key": spec.inchikey14,
                         "smiles": spec.normalized_smiles})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(tc)} structures ({time.time() - t0:.0f}s), "
                  f"{len(rows):,} unexplained peaks", flush=True)

    print(f"\nunexplained intensity fraction (permissive theory): "
          f"{unexplained_intensity / max(1e-9, total_intensity):.4f} "
          f"over {total_peaks:,} peaks", flush=True)

    df = pd.DataFrame(rows)
    df["bin"] = (df["mz"] * 100).round() / 100.0
    agg = (df.groupby("bin")
           .agg(n=("mz", "size"), mean_share=("share", "mean"),
                total_share=("share", "sum"), mz_mean=("mz", "mean"),
                mz_spread_ppm=("mz", lambda s: (s.max() - s.min())
                                / max(1e-9, s.mean()) * 1e6))
           .reset_index()
           .sort_values(["n", "mean_share"], ascending=False))
    agg["score"] = agg["n"] * agg["mean_share"]
    agg = agg.sort_values("score", ascending=False)
    agg.to_csv(args.out, index=False)
    print(f"saved {len(agg):,} peak bins -> {args.out}\n", flush=True)

    print("top unexplained m/z bins (count x mean intensity share):", flush=True)
    top = agg.head(40).copy()
    for _, r in top.iterrows():
        formulas = assign_formulas(r["mz_mean"], 5.0)
        fs = " | ".join(formulas[:3]) if formulas else ""
        print(f"  {r['mz_mean']:9.4f}  n={int(r['n']):5d}  share={r['mean_share']:.3f}"
              f"  spread={r['mz_spread_ppm']:.0f}ppm  {fs}", flush=True)


def assign_formulas(mz: float, tol_ppm: float) -> list[str]:
    """Elemental compositions for the ion (incl. charge carrier) within tolerance."""
    els = [("C", 12.0, 40), ("H", 1.00782503223, 80), ("N", 14.00307400443, 12),
           ("O", 15.99491461957, 18), ("S", 31.97207117442, 4),
           ("P", 30.97376199842, 4)]
    target = mz - PROTON_MASS
    out = []
    from itertools import product
    c_n, h_n = int(target / 12) + 2, int(target / 1.00783) + 2
    for c in range(max(0, c_n - 2), c_n):
        rem_c = target - c * 12.0
        if rem_c < -0.02:
            break
        for n in range(0, 13):
            rem_n = rem_c - n * 14.00307400443
            if rem_n < -0.02:
                break
            for o in range(0, 19):
                rem_o = rem_n - o * 15.99491461957
                if rem_o < -0.02:
                    break
                for s in range(0, 5):
                    rem_s = rem_o - s * 31.97207117442
                    if rem_s < -0.02:
                        break
                    for p in range(0, 5):
                        rem_p = rem_s - p * 30.97376199842
                        if rem_p < -0.02:
                            break
                        h_lo = max(0, int((rem_p - 0.02) / 1.00783))
                        h_hi = min(80, int((rem_p + 0.02) / 1.00783))
                        for h in range(h_lo, h_hi + 1):
                            mass = rem_p - h * 1.00782503223
                            ppm = mass / target * 1e6 if target else 999
                            if abs(ppm) <= tol_ppm:
                                f = f"C{c}H{h}" + (f"N{n}" if n else "") \
                                    + (f"O{o}" if o else "") + (f"S{s}" if s else "") \
                                    + (f"P{p}" if p else "")
                                rdb = c - h / 2 + n / 2 + p / 2 + 1
                                if 0 <= rdb <= 20 and h <= 2 * c + n + 2:
                                    out.append(f"{f} [M+H]+ {ppm:+.1f}ppm rdb{rdb:.1f}")
    return out[:6]


if __name__ == "__main__":
    main()
