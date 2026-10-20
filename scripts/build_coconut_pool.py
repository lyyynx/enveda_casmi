"""Build the COCONUT pool asset for the kernel and Kaggle dataset upload.

Canonicalizes COCONUT SMILES, verifies inchikey14 and neutral masses, and
writes a compact parquet (inchikey14, smiles, mass) for runtime use.

Usage:
    uv run python scripts/build_coconut_pool.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    meta = pickle.load(open(PROJECT_ROOT / "data/coconut/coco_meta.pkl", "rb"))
    keys = [str(k)[:14] for k in meta["keys"]]
    smis = [str(s) for s in meta["smiles"]]
    masses = np.load(PROJECT_ROOT / "data/coconut/coco_mass.npy")
    print(f"coconut: {len(keys):,} structures")

    # canonicalize + verify in a single pass
    rows = []
    bad = 0
    mass_dev = []
    for i, (k, s) in enumerate(zip(keys, smis)):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            bad += 1
            continue
        can = Chem.MolToSmiles(mol)
        ik = Chem.MolToInchiKey(mol)
        if ik is None:
            bad += 1
            continue
        m = Descriptors.ExactMolWt(mol)
        mass_dev.append(abs(m - masses[i]))
        rows.append((ik[:14], can, m))
        if (i + 1) % 100_000 == 0:
            print(f"  {i + 1:,} processed", flush=True)

    print(f"parsed: {len(rows):,}  invalid: {bad:,}")
    mass_dev = np.asarray(mass_dev)
    print(f"mass consistency vs coco_mass.npy: max {mass_dev.max():.6f}  "
          f"mean {mass_dev.mean():.6f}  >0.001 Da: {(mass_dev > 0.001).sum()}")

    df = pd.DataFrame(rows, columns=["inchikey14", "smiles", "mass"])
    before = len(df)
    df = df.drop_duplicates("inchikey14", keep="first")
    print(f"unique inchikey14: {len(df):,} (dropped {before - len(df):,} tautomer/merge dupes)")

    train = pd.read_parquet(PROJECT_ROOT / "data/train.parquet",
                            columns=["inchikey14"])
    train_keys = set(train["inchikey14"].dropna())
    overlap = df["inchikey14"].isin(train_keys)
    print(f"overlap with train structures: {overlap.sum():,}")

    out = PROJECT_ROOT / "data/coconut_pool.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
