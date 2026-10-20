"""Inspect the strict-vs-formula gap molecules: truth vs our #1 structure."""
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

res = pd.read_csv(PROJECT_ROOT / "benchmark" / "rescore_tautomer.csv")
mapping = pd.read_csv(PROJECT_ROOT / "data" / "cut_map.csv")
truth_smiles = dict(zip(mapping["molecule_id"], mapping["normalized_smiles"]))
sub = pd.read_csv(Path(r"C:\Users\Michael\AppData\Local\Temp\opencode\kernel_validate\submission.csv"))
sub_smiles = dict(zip(sub["molecule_id"], sub["smiles"]))

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(s):
    m = Chem.MolFromSmiles(s)
    return gen.GetFingerprint(m) if m else None


def tan(a, b):
    fa, fb = fp(a), fp(b)
    if fa is None or fb is None:
        return -1.0
    return DataStructs.TanimotoSimilarity(fa, fb)


gap = res[(res["rr_strict"] < res["rr_formula"])].copy()
gap = gap.sort_values("formula")
print(f"gap molecules (formula rank < strict rank): {len(gap)}\n")

n = 0
for _, r in gap.iterrows():
    mid = r["molecule_id"]
    slots = str(sub_smiles[mid]).split(";")
    t = truth_smiles[mid]
    if r["formula"] != 1:
        continue
    n += 1
    if n > 10:
        break
    top = slots[0]
    tan_t = tan(t, top)
    strict_rank = r["strict"]
    print(f"{mid}  Tanimoto(truth, #1)={tan_t:.3f}  strict_rank={strict_rank}")
    print(f"  truth : {t}")
    print(f"  #1    : {top}")
    where = next((i for i, s in enumerate(slots, 1)
                  if (Chem.MolToInchiKey(Chem.MolFromSmiles(s)) or "")[:14]
                  == (Chem.MolToInchiKey(Chem.MolFromSmiles(t)) or "")[:14]), 0)
    print(f"  truth's own rank in our 25: {where}")
    print()
