"""Carve a realistic local validation split out of train.parquet.

Holds out entire structures (all their spectra) so the validation can never
leak, and produces:

  data/train_cut.parquet       train minus ALL rows of held-out structures
                               (drop-in replacement for train.parquet)
  data/test_cut.parquet        held-out spectra grouped into molecules
                               (schema mirrors test.parquet + truth columns)
  data/pool_extra.parquet      held-out structures as structure-only entries
                               (they play the role of external-DB candidates,
                               exactly like COCONUT entries for Class 2)
  data/val_external_pool.parquet  coconut_pool + pool_extra (validation pool)
  data/cut_map.csv             molecule_id -> truth key/smiles/slice

Slices:
  normal   structures with >=1 clean spectrum (mirrors the hidden test)
  defect   structures whose every spectrum fails cleaning (minor defects;
           the runtime cleaner may still keep 1-4 peaks)

Usage:
    uv run python scripts/make_split.py [--n-normal 400] [--n-defect 60]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from enveda_casmi26.pipeline import neutral_mass  # noqa: E402

DATA = PROJECT_ROOT / "data"
TRAIN = DATA / "train.parquet"
TRAIN_CLEAN = DATA / "train_clean.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-normal", type=int, default=400)
    parser.add_argument("--n-defect", type=int, default=60)
    parser.add_argument("--max-spectra", type=int, default=4,
                        help="spectra per test molecule (hidden test avg ~3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------- structure pools
    tc = pq.read_table(TRAIN_CLEAN, columns=["inchikey14", "drop_reason"]).to_pandas()
    n_train = pq.ParquetFile(TRAIN).metadata.num_rows
    assert len(tc) == n_train, "train_clean rows do not align with train.parquet"
    keys = tc["inchikey14"].to_numpy(dtype=object)
    clean_keys = set(pd.unique(tc.loc[tc["drop_reason"] == "", "inchikey14"]))
    all_keys = set(pd.unique(keys))
    defect_keys = all_keys - clean_keys
    print(f"structures: {len(all_keys):,} total, {len(clean_keys):,} with clean "
          f"spectra, {len(defect_keys):,} defect-only")

    hold_normal = set(rng.choice(sorted(k for k in clean_keys if k),
                                 size=min(args.n_normal, len(clean_keys)),
                                 replace=False).tolist())
    # defect structures: need >=1 spectrum with at least 3 raw peaks to be a
    # remotely usable query
    pf0 = pq.ParquetFile(TRAIN)
    defect_ok = {}
    for batch in pf0.iter_batches(batch_size=500_000,
                                  columns=["inchikey14", "num_peaks"]):
        for k, n in zip(batch.column("inchikey14").to_pylist(),
                        batch.column("num_peaks").to_pylist()):
            if k in defect_keys and n and n >= 3:
                defect_ok[k] = max(defect_ok.get(k, 0), n)
    hold_defect = set(rng.choice(sorted(defect_ok), size=min(args.n_defect,
                                                            len(defect_ok)),
                                 replace=False).tolist())
    held = hold_normal | hold_defect
    print(f"holding out: {len(hold_normal)} normal + {len(hold_defect)} defect "
          f"structures ({len(held)} total)")

    # ---------------------------------------- pass 1: write train_cut, collect held rows
    print("writing train_cut.parquet ...")
    pf = pq.ParquetFile(TRAIN)
    schema = pf.schema_arrow
    dra = tc["drop_reason"].to_numpy()
    held_rows = []
    held_reasons = []
    out = pq.ParquetWriter(DATA / "train_cut.parquet", schema)
    n_written = 0
    row0 = 0
    for batch in pf.iter_batches(batch_size=200_000):
        bk = batch.column("inchikey14").to_pylist()
        mask = [k not in held for k in bk]
        if not all(mask):
            held_rows.append(pa.Table.from_batches(
                [batch.filter([not m for m in mask])]))
            held_reasons.extend(dra[row0:row0 + len(bk)][[i for i, m in
                                                          enumerate(mask) if not m]])
        if any(mask):
            out.write_batch(batch.filter(mask))
            n_written += sum(mask)
        row0 += len(bk)
        if n_written and (n_written // 200_000) != ((n_written - sum(mask)) // 200_000):
            print(f"  {n_written:,} rows written", flush=True)
    out.close()
    print(f"train_cut: {n_written:,} rows")

    held_tab = pa.concat_tables(held_rows)
    hd = held_tab.to_pandas()
    hd["drop_reason"] = held_reasons
    print(f"held-out rows: {len(hd):,}")

    # ---------------------------------------- molecules
    rows = []
    mapping = []
    n_id = 0
    for slice_name, keyset in (("normal", hold_normal), ("defect", hold_defect)):
        for k in sorted(keyset):
            sub = hd[hd["inchikey14"] == k]
            if slice_name == "normal":
                sub = sub[sub["drop_reason"] == ""]
            if sub.empty:
                sub = hd[hd["inchikey14"] == k]  # defect: take whatever exists
            n_spec = min(len(sub), rng.integers(1, args.max_spectra + 1))
            picks = sub.sample(n=n_spec, random_state=int(rng.integers(1 << 31)))
            mid = f"cut_{n_id:04d}"
            n_id += 1
            for _, r in picks.iterrows():
                rows.append({
                    "molecule_id": mid,
                    "ms2_mzs": r["ms2_mzs"],
                    "ms2_normalized_intensities": r["ms2_normalized_intensities"],
                    "precursor_mz": r["precursor_mz"],
                    "adduct": r["adduct"],
                    "ionization_mode": r["ionization_mode"],
                    "instrument_type": r["instrument_type"],
                    "collision_energy_orig": r["collision_energy_orig"],
                    "collision_energy_orig_units": r["collision_energy_orig_units"],
                    "inchikey14": k,
                    "normalized_smiles": r["normalized_smiles"],
                    "slice": slice_name,
                })
            mapping.append({"molecule_id": mid, "inchikey14": k,
                            "normalized_smiles": picks.iloc[0]["normalized_smiles"],
                            "slice": slice_name})

    test_cut = pd.DataFrame(rows)
    test_cut.to_parquet(DATA / "test_cut.parquet", index=False)
    pd.DataFrame(mapping).to_csv(DATA / "cut_map.csv", index=False)
    print(f"test_cut: {len(test_cut)} spectra, {len(mapping)} molecules "
          f"({test_cut['slice'].value_counts().to_dict()})")

    # ---------------------------------------- pool_extra: held structures as external DB
    hd2 = held_tab.select(["inchikey14", "normalized_smiles", "precursor_mz",
                           "adduct"]).to_pandas()
    hd2["nm"] = [neutral_mass(p, a) for p, a in zip(hd2["precursor_mz"], hd2["adduct"])]
    extra = (hd2.groupby("inchikey14")
             .agg(smiles=("normalized_smiles", "first"), mass=("nm", "median"))
             .reset_index()
             .rename(columns={"inchikey14": "inchikey14"}))
    extra = extra[extra["mass"].notna()]
    extra.to_parquet(DATA / "pool_extra.parquet", index=False)
    print(f"pool_extra: {len(extra):,} structure-only entries")

    coconut = pd.read_parquet(DATA / "coconut_pool.parquet")
    val_pool = pd.concat([coconut, extra.rename(columns={"smiles": "smiles"})],
                         ignore_index=True).drop_duplicates("inchikey14")
    val_pool.to_parquet(DATA / "val_external_pool.parquet", index=False)
    print(f"val_external_pool: {len(val_pool):,} structures "
          f"(coconut {len(coconut):,} + extra {len(extra):,})")


if __name__ == "__main__":
    main()
