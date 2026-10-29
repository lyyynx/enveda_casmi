"""Build exact-mass table for train structures + fix val_external_pool masses.

Pool mass conventions must be uniform (RDKit monoisotopic from SMILES):
  - train structures: kernel currently uses median precursor-derived masses
    (carries instrument calibration error per structure)
  - val_external_pool holdout rows: precursor-derived (same problem)
  - coconut_pool: already exact

Outputs:
  kaggle_assets/dataset/train_exact_masses.parquet  (inchikey14, neutral_mass)
  data/val_external_pool.parquet                    (mass column recomputed)
"""
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

t0 = time.time()
train = pd.read_parquet(PROJECT_ROOT / "data" / "train_cut.parquet",
                         columns=["inchikey14", "normalized_smiles"])
u = train.dropna(subset=["normalized_smiles"]).drop_duplicates("inchikey14")
print(f"unique train structures: {len(u):,}")

masses = []
smis = u["normalized_smiles"].tolist()
keys = u["inchikey14"].tolist()
cache: dict[str, float] = {}
for i, s in enumerate(smis):
    m = cache.get(s)
    if m is None:
        mol = Chem.MolFromSmiles(s)
        m = rdMolDescriptors.CalcExactMolWt(mol) if mol else float("nan")
        cache[s] = m
    masses.append(m)
out = pd.DataFrame({"inchikey14": keys, "neutral_mass": masses}).dropna()
out.to_parquet(PROJECT_ROOT / "kaggle_assets" / "dataset" / "train_exact_masses.parquet",
               index=False)
print(f"train exact masses: {len(out):,} rows ({time.time() - t0:.0f}s)")

val_path = PROJECT_ROOT / "data" / "val_external_pool.parquet"
val = pd.read_parquet(val_path)
col_mass = "mass" if "mass" in val.columns else "neutral_mass"
col_smi = "smiles" if "smiles" in val.columns else "normalized_smiles"
fixed = []
for s in val[col_smi]:
    mol = Chem.MolFromSmiles(s)
    fixed.append(rdMolDescriptors.CalcExactMolWt(mol) if mol else float("nan"))
val[col_mass] = fixed
val = val.dropna(subset=[col_mass])
val.to_parquet(val_path, index=False)
n_bad = int((pd.Series(fixed) - pd.read_parquet(PROJECT_ROOT / "data" / "val_external_pool.parquet")[col_mass]).abs().gt(1e-9).sum())
print(f"val_external_pool: recomputed {len(val):,} masses, changed {n_bad}")
print(f"done ({time.time() - t0:.0f}s)")
