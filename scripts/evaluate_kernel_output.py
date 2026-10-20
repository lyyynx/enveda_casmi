"""Score a kernel submission.csv against the placeholder test's leaked truth.

The public test.parquet spectra are exact duplicates of train rows, so the
truth inchikey14 per molecule is known. This measures the kernel pipeline's
end-to-end quality the same way the competition does (MRR@25 on inchikey14).

Usage:
    uv run python scripts/evaluate_kernel_output.py benchmark/kernel_v5_out/submission.csv
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"


def spec_key(mzs, ints) -> str:
    m = np.asarray(mzs, dtype=np.float64)
    i = np.asarray(ints, dtype=np.float64)
    return hashlib.md5(m.tobytes() + i.tobytes()).hexdigest()


def main() -> None:
    sub_path = Path(sys.argv[1])
    sub = pd.read_csv(sub_path)

    print("hashing train spectra ...")
    pf = pq.ParquetFile(DATA / "train.parquet")
    hash2key = {}
    for batch in pf.iter_batches(batch_size=100_000,
                                 columns=["inchikey14", "ms2_mzs",
                                          "ms2_normalized_intensities"]):
        for k, m, i in zip(batch.column("inchikey14").to_pylist(),
                            batch.column("ms2_mzs").to_pylist(),
                            batch.column("ms2_normalized_intensities").to_pylist()):
            if k and m:
                hash2key[spec_key(m, i)] = k
    print(f"  {len(hash2key):,} unique spectra")

    test = pq.read_table(DATA / "test.parquet",
                         columns=["molecule_id", "ms2_mzs",
                                  "ms2_normalized_intensities"]).to_pandas()
    mol_keys = {}
    for mid, m, i in zip(test["molecule_id"], test["ms2_mzs"],
                         test["ms2_normalized_intensities"]):
        k = hash2key.get(spec_key(m, i))
        if k:
            mol_keys.setdefault(mid, []).append(k)

    truth = {}
    conflicts = 0
    for mid, keys in mol_keys.items():
        counts = pd.Series(keys).value_counts()
        truth[mid] = counts.index[0][:14]
        if len(counts) > 1:
            conflicts += 1
    print(f"truth resolved for {len(truth)}/{len(mol_keys)} molecules "
          f"({conflicts} with conflicting spectra)")

    cache = {}


    def ikey14(smiles: str) -> str | None:
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles)
            ik = Chem.MolToInchiKey(mol) if mol else None
            cache[smiles] = ik[:14] if ik else None
        return cache[smiles]

    rrs, top1, solved, missing = [], 0, 0, 0
    for _, r in sub.iterrows():
        mid = r["molecule_id"]
        t = truth.get(mid)
        if t is None:
            missing += 1
            continue
        keys = [ikey14(s) for s in str(r["smiles"]).split(";")]
        rr = 0.0
        for rank, k in enumerate(keys[:25], start=1):
            if k == t:
                rr = 1.0 / rank
                break
        rrs.append(rr)
        solved += rr > 0
        top1 += keys[:1] == [t]

    n = len(rrs)
    print(f"\n{sub_path.name}: {n} scored molecules ({missing} without truth)")
    print(f"MRR@25 : {np.mean(rrs):.4f}")
    print(f"top-1  : {top1/n:.3f}   any-rank: {solved/n:.3f}")


if __name__ == "__main__":
    main()
