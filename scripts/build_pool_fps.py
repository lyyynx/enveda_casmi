"""Precompute Morgan fingerprints for the entire candidate pool.

Covers every structure the kernel can ever propose: full train.parquet
structures (including enveda-180 and holdout), COCONUT, and the external
validation pool (held-out truths). Output: kaggle_assets/dataset/pool_fps.npz
with 'keys' (U14) and 'fps' (packed uint8 bits, N x 256).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
N_BITS = 2048


def main():
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)

    print("loading pool sources ...")
    tr = pd.read_parquet(r"data\train.parquet",
                         columns=["inchikey14", "normalized_smiles"])
    tr = tr.dropna(subset=["normalized_smiles"]).drop_duplicates(subset="inchikey14")
    tr = tr.rename(columns={"normalized_smiles": "smiles"})[["inchikey14", "smiles"]]
    coco = pd.read_parquet(r"data\coconut_pool.parquet")[["inchikey14", "smiles"]]
    ext = pd.read_parquet(r"data\val_external_pool.parquet")[["inchikey14", "smiles"]]
    pool = pd.concat([tr, coco, ext], ignore_index=True)
    pool = pool.dropna(subset=["smiles"]).drop_duplicates(subset="inchikey14")
    pool = pool[pool.inchikey14.str.len() == 14]
    print(f"pool structures: {len(pool):,}")

    keys = []
    bits = np.zeros((len(pool), N_BITS // 8), dtype=np.uint8)
    t0 = time.time()
    n_bad = 0
    for i, r in enumerate(pool.itertuples(index=False)):
        mol = Chem.MolFromSmiles(r.smiles)
        if mol is None:
            n_bad += 1
            continue
        fp = fp_gen.GetFingerprint(mol)
        row = np.zeros(N_BITS, dtype=bool)
        for b in fp.GetOnBits():
            row[b] = True
        keys.append(r.inchikey14)
        bits[len(keys) - 1] = np.packbits(row)
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1}/{len(pool)} ({time.time() - t0:.0f}s)", flush=True)
    bits = bits[: len(keys)]
    print(f"fingerprinted {len(keys):,} structures ({n_bad} unparseable) "
          f"in {time.time() - t0:.0f}s")

    out = r"kaggle_assets\dataset\pool_fps.npz"
    np.savez(out, keys=np.array(keys, dtype="<U14"), fps=bits)
    print(f"saved -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
