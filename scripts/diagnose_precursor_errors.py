"""Diagnose precursor m/z errors against theoretical masses (truth SMILES).

For every train spectrum: parse the adduct label generically (molecule count,
atom delta, charge), compute the theoretical precursor m/z from the exact
monoisotopic mass of the truth structure, and characterize the error:

  * global magnitude distribution (ppm buckets),
  * spike attribution: wrong isotope picked, adduct swaps, in-source losses,
  * dimer adducts ([2M+H]+ ...): does precursor_mz match the dimer ion or the
    monomer ion? (ADDUCT_ALIASES collapses dimers to monomers in the pipeline),
  * ppm drift vs m/z (calibration) and per-instrument breakdown,
  * cross-check with the provided precursor_error_ppm column.

Usage:
    uv run python scripts/diagnose_precursor_errors.py [--sample 15]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from enveda_casmi26.fragmentation import parse_adduct_label as parse_adduct  # noqa: E402

# adduct-swap offsets: err_da if the *true* ion was `true` but labeled `labeled`
SWAPS = [
    ("true [M+Na]+, labeled [M+H]+", 22.989220 - 1.007276),
    ("true [M+K]+, labeled [M+H]+", 38.963158 - 1.007276),
    ("true [M+NH4]+, labeled [M+H]+", 18.033823 - 1.007276),
    ("true [M+H]+, labeled [M+Na]+", -(22.989220 - 1.007276)),
    ("true [M+H]+, labeled [M+K]+", -(38.963158 - 1.007276)),
    ("true [M+Cl]-, labeled [M-H]-", 34.969402 + 1.007276),
    ("true [M-H]-, labeled [M+Cl]-", -(34.969402 + 1.007276)),
]
LOSSES = [
    ("in-source H2O loss", -18.0105646),
    ("in-source NH3 loss", -17.0265481),
]


def exact_masses(smiles_iter):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    out, bad = {}, 0
    for s in smiles_iter:
        if s in out:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            bad += 1
            out[s] = np.nan
        else:
            out[s] = rdMolDescriptors.CalcExactMolWt(mol)
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="print N example outliers")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train_cut.parquet"))
    args = ap.parse_args()

    df = pd.read_parquet(args.data,
                         columns=["inchikey14", "normalized_smiles", "precursor_mz",
                                  "adduct", "ionization_mode", "instrument_type",
                                  "precursor_error_ppm"])
    print(f"rows: {len(df)}")

    parsed = df.adduct.map(parse_adduct)
    idx_ok = parsed.notna()
    dfp = df[idx_ok].copy()
    print(f"adduct parsed: {len(dfp)} rows ({len(df) - len(dfp)} unparseable)")

    masses, bad_smiles = exact_masses(dfp.normalized_smiles.unique())
    print(f"unique structures: {len(masses)} ({bad_smiles} unparseable SMILES)")

    dfp["n_mol"] = [p[0] for p in parsed[idx_ok]]
    dfp["delta"] = [p[1] for p in parsed[idx_ok]]
    dfp["z"] = [p[2] for p in parsed[idx_ok]]
    dfp["M"] = dfp.normalized_smiles.map(masses)
    dfp = dfp[dfp.M.notna() & np.isfinite(dfp.precursor_mz)]

    dfp["theo"] = (dfp.n_mol * dfp.M + dfp.delta) / dfp.z
    dfp["err_da"] = dfp.precursor_mz - dfp.theo
    dfp["err_ppm"] = dfp.err_da / dfp.theo * 1e6

    print("\n=== global magnitude ===")
    a = dfp.err_ppm.abs()
    for lab, cond in [
        ("|err| < 5 ppm", a < 5), ("|err| < 10 ppm", a < 10),
        ("|err| < 20 ppm", a < 20), ("|err| < 50 ppm", a < 50),
        ("|err| < 0.02 Da", dfp.err_da.abs() < 0.02),
        ("0.02 <= |err| < 0.5 Da", (dfp.err_da.abs() >= 0.02) & (dfp.err_da.abs() < 0.5)),
        ("0.5 <= |err| < 2 Da", (dfp.err_da.abs() >= 0.5) & (dfp.err_da.abs() < 2)),
        ("|err| >= 2 Da", dfp.err_da.abs() >= 2),
    ]:
        print(f"  {lab:<24} {cond.mean() * 100:6.2f}%  ({cond.sum()})")
    print(f"  median |err|: {a.median():.2f} ppm")

    print("\n=== outlier attribution (|err| >= 0.05 Da) ===")
    out = dfp[dfp.err_da.abs() >= 0.05]
    print(f"outliers: {len(out)} rows ({len(out) / len(dfp) * 100:.2f}%)")
    tol = 0.02 / out.z
    hypo = pd.Series("unexplained", index=out.index)
    hypo[(out.err_da.abs() <= 0.02)] = "measurement/noise"
    for name, off in [("isotope M+1 picked", 1.00335), ("isotope M+2 picked", 2.00670)]:
        res = (out.err_da - off / out.z).abs()
        take = (res <= tol) & (hypo == "unexplained")
        hypo[take] = name
    for name, off in SWAPS:
        take = ((out.err_da - off).abs() <= 0.02) & (hypo == "unexplained")
        hypo[take] = name
    for name, off in LOSSES:
        take = ((out.err_da - off).abs() <= 0.02) & (hypo == "unexplained")
        hypo[take] = name
    # dimer hypothesis for labeled monomers: does err ~ M (i.e. true n=2)?
    mono = out[out.n_mol == 1]
    take = (mono.err_da - mono.M).abs() <= 0.05
    hypo[take[take].index] = "true [2M+adduct]+, labeled monomer"
    print(hypo.value_counts().to_string())

    print("\n=== dimer-labeled rows ([nM+...], n>1): dimer vs monomer fit ===")
    dim = dfp[dfp.n_mol > 1]
    theo_mono = (dim.M + dim.delta) / dim.z
    fit_dimer = (dim.precursor_mz - dim.theo).abs() < 0.02
    fit_mono = (dim.precursor_mz - theo_mono).abs() < 0.02
    print(f"  rows: {len(dim)}")
    print(f"  precursor matches DIMER ion (n as labeled): {fit_dimer.mean() * 100:.1f}%")
    print(f"  precursor matches MONOMER ion:               {fit_mono.mean() * 100:.1f}%")
    print(f"  matches neither:                             {(~(fit_dimer | fit_mono)).mean() * 100:.1f}%")
    for ad, g in dim.groupby("adduct"):
        if len(g) < 1000:
            continue
        fd = (g.precursor_mz - g.theo).abs() < 0.02
        fm = (g.precursor_mz - theo_mono[g.index]).abs() < 0.02
        print(f"    {ad:<12} n={len(g):>7}  dimer {fd.mean() * 100:5.1f}%  monomer {fm.mean() * 100:5.1f}%")

    print("\n=== calibration drift: ppm error vs m/z (|err| < 0.05 Da) ===")
    ins = dfp[dfp.err_da.abs() < 0.05]
    r = np.corrcoef(ins.theo, ins.err_ppm)[0, 1]
    print(f"  pearson(mz, err_ppm) = {r:+.3f} over {len(ins)} rows")
    qs = pd.qcut(ins.theo, 8)
    med = ins.groupby(qs, observed=True).agg(mz=("theo", "median"),
                                              med_ppm=("err_ppm", lambda x: x.abs().median()))
    for _, row in med.iterrows():
        print(f"    mz~{row.mz:7.1f}  median|ppm| {row.med_ppm:6.1f}")

    print("\n=== per instrument_type ===")
    g = dfp.groupby("instrument_type").agg(
        n=("err_da", "size"),
        med_abs_ppm=("err_ppm", lambda x: x.abs().median()),
        outlier_pct=("err_da", lambda x: (x.abs() >= 0.05).mean() * 100),
    ).sort_values("n", ascending=False)
    print(g.head(12).to_string())

    print("\n=== per adduct (top 12 by count) ===")
    g = dfp.groupby("adduct").agg(
        n=("err_da", "size"),
        med_abs_ppm=("err_ppm", lambda x: x.abs().median()),
        outlier_pct=("err_da", lambda x: (x.abs() >= 0.05).mean() * 100),
    ).sort_values("n", ascending=False)
    print(g.head(12).to_string())

    print("\n=== provided precursor_error_ppm column ===")
    have = dfp[dfp.precursor_error_ppm.notna()]
    print(f"  rows with value: {len(have)}")
    if len(have):
        print(f"  median |value|: {have.precursor_error_ppm.abs().median():.2f}")
        r = np.corrcoef(have.precursor_error_ppm, have.err_ppm)[0, 1]
        print(f"  pearson(col, my err_ppm) = {r:+.3f}")

    if args.sample:
        cols = ["inchikey14", "adduct", "precursor_mz", "theo", "err_da", "err_ppm"]
        print(f"\n=== sample of {args.sample} largest outliers ===")
        print(dfp.nlargest(args.sample, "err_da")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
