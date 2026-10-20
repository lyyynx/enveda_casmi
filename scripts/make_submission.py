"""Build a competition submission from the similarity baseline.

Per test molecule:
  1. rank train structures by best cosine similarity across the molecule's
     spectra (library search, see similarity_baseline.py)
  2. fill remaining slots (up to 25) with structures whose neutral mass is
     closest to the test molecule's neutral mass

Usage:
    uv run python scripts/make_submission.py --out submissions/submission_v0.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enveda_casmi26.fragmentation import ADDUCTS, ADDUCT_ALIASES
from similarity_baseline import Library

DATA_DIR = PROJECT_ROOT / "data"
N_SLOTS = 25


def neutral_mass(precursor_mz: float, adduct: str):
    adduct = ADDUCT_ALIASES.get(adduct, adduct)
    if adduct not in ADDUCTS or not np.isfinite(precursor_mz):
        return np.nan
    delta, charge = ADDUCTS[adduct]
    return precursor_mz * charge - delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="submissions/submission_v0.csv")
    args = parser.parse_args()

    print("Building structure library ...")
    lib = Library()

    print("Computing library neutral masses ...")
    lib_neutral = np.array([neutral_mass(mz, ad) for mz, ad in
                            zip(lib.reps["precursor_mz"], lib.reps["adduct"])])
    lib_smiles = lib.smiles
    valid = np.isfinite(lib_neutral)

    test = pd.read_parquet(DATA_DIR / "test_clean.parquet")

    rows = []
    n_padded = 0
    for mol, group in test.groupby("molecule_id"):
        best_per_structure: dict[str, float] = {}
        for spec in group.itertuples(index=False):
            hits = lib.search(spec.ms2_mzs, spec.ms2_normalized_intensities,
                              spec.ionization_mode, top_k=N_SLOTS)
            for idx, score in hits:
                key = lib.keys[idx]
                best_per_structure[key] = max(best_per_structure.get(key, 0.0), score)
        ranked = sorted(best_per_structure.items(), key=lambda t: -t[1])
        smiles_list = [lib.smiles[np.where(lib.keys == k)[0][0]] for k, _ in ranked]

        if len(smiles_list) < N_SLOTS:
            n_padded += 1
            q_neutral = np.nanmedian([neutral_mass(s.precursor_mz, s.adduct)
                                      for s in group.itertuples(index=False)])
            if np.isfinite(q_neutral):
                order = np.argsort(np.abs(lib_neutral - q_neutral))
                for i in order:
                    if len(smiles_list) >= N_SLOTS:
                        break
                    if not valid[i]:
                        continue
                    s = lib_smiles[i]
                    if s not in smiles_list:
                        smiles_list.append(s)
        if len(smiles_list) < N_SLOTS:
            top_smiles = lib_smiles[:N_SLOTS]
            for s in top_smiles:
                if len(smiles_list) >= N_SLOTS:
                    break
                if s not in smiles_list:
                    smiles_list.append(s)

        rows.append({"molecule_id": mol, "smiles": ";".join(smiles_list[:N_SLOTS])})

    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)

    n_slots = out["smiles"].str.count(";") + 1
    print(f"\nwrote {out_path} ({len(out)} molecules)")
    print(f"  slot count: min {n_slots.min()} / max {n_slots.max()}")
    print(f"  molecules padded with mass candidates: {n_padded}")
    example = out.iloc[0]
    print(f"  example ({example['molecule_id']}): {example['smiles'][:120]} ...")


if __name__ == "__main__":
    main()
