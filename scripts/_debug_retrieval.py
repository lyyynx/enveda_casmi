"""Debug candidate retrieval for test_cut molecules."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from enveda_casmi26.fragmentation import ADDUCTS, ADDUCT_ALIASES  # noqa: E402
from enveda_casmi26.pipeline import CandidatePool, Molecule, MoleculeSpectrum, neutral_mass  # noqa: E402

# pool: train_cut structures + val external pool (like the kernel)
tab = pq.read_table(ROOT / "data/train_cut.parquet",
                    columns=["inchikey14", "normalized_smiles", "precursor_mz",
                             "adduct"]).to_pandas()
tab = tab[tab["inchikey14"].notna()]
tab["nm"] = [neutral_mass(m, a) for m, a in zip(tab["precursor_mz"], tab["adduct"])]
structs = (tab.groupby("inchikey14")
           .agg(normalized_smiles=("normalized_smiles", "first"),
                neutral_mass=("nm", "median")).reset_index())
coco = pd.read_parquet(ROOT / "data/val_external_pool.parquet")
coco = coco.rename(columns={"smiles": "normalized_smiles", "mass": "neutral_mass"})
pool = CandidatePool(pd.concat([structs, coco], ignore_index=True)
                     .drop_duplicates("inchikey14", keep="first"))
print(f"pool: {len(pool):,}  mass range {pool.masses.min():.1f}..{pool.masses.max():.1f}")

test = pd.read_parquet(ROOT / "data/test_cut.parquet")
mapping = pd.read_csv(ROOT / "data/cut_map.csv")
truth_of = dict(zip(mapping["molecule_id"], mapping["inchikey14"]))

mols = {}
for r in test.itertuples(index=False):
    m = mols.setdefault(r.molecule_id, Molecule(r.molecule_id))
    m.spectra.append(MoleculeSpectrum(np.asarray(r.ms2_mzs), np.asarray(r.ms2_normalized_intensities),
                                      r.adduct, float(r.precursor_mz), r.ionization_mode))

rng = np.random.default_rng(0)
ids = [m for m in mols if mapping.set_index("molecule_id").loc[m, "slice"] == "normal"]
for mid in ids[:8]:
    mol = mols[mid]
    qm = mol.neutral_mass
    tk = truth_of[mid]
    ti = pool.key_to_idx.get(tk)
    info = f"{mid}: query_mass={qm:.4f}"
    if np.isfinite(qm):
        for ppm in (8.5, 30.0, 100.0):
            w = pool.window(qm, ppm)
            info += f"  w{ppm}={len(w)}"
    else:
        info += "  QUERY MASS NaN"
    if ti is None:
        info += "  truth NOT IN POOL"
    else:
        d = abs(pool.masses[ti] - qm) if np.isfinite(qm) else float("nan")
        info += (f"  truth_mass={pool.masses[ti]:.4f} delta={d * 1e6 / qm:.1f}ppm"
                 if np.isfinite(qm) else f"  truth_mass={pool.masses[ti]:.4f}")
        spec = [(s.precursor_mz, s.adduct) for s in mol.spectra]
        info += f"  spectra={spec}"
    print(info)
