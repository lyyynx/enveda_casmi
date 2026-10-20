"""Tautomer-aware rescoring of the validation submission.

For every test_cut molecule, compares the submitted structures to the truth
with three levels of strictness:
  strict      inchikey14 equality (what we have been measuring)
  formula     same molecular formula (cheap tautomer proxy)
  tautomer    equality after RDKit tautomer canonicalization (both sides)

This separates real retrieval failures from representation mismatches.

Usage:
    uv run python scripts/rescore_tautomer.py
"""

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SUB = Path(r"C:\Users\Michael\AppData\Local\Temp\opencode\kernel_validate\submission.csv")


def main() -> None:
    sub = pd.read_csv(SUB)
    mapping = pd.read_csv(PROJECT_ROOT / "data" / "cut_map.csv")
    truth_of = dict(zip(mapping["molecule_id"], mapping["inchikey14"]))
    smiles_of = dict(zip(mapping["molecule_id"], mapping["normalized_smiles"]))

    enum = rdMolStandardize.TautomerEnumerator()

    cache_t = {}
    cache_f = {}

    def taut_key(smiles: str):
        if smiles not in cache_t:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                cache_t[smiles] = None
            else:
                k = Chem.MolToInchiKey(enum.Canonicalize(m))
                cache_t[smiles] = k[:14] if k else None
        return cache_t[smiles]

    def formula(smiles: str):
        if smiles not in cache_f:
            m = Chem.MolFromSmiles(smiles)
            cache_f[smiles] = rdMolDescriptors.CalcMolFormula(m) if m else None
        return cache_f[smiles]

    rows = []
    for _, r in sub.iterrows():
        mid = r["molecule_id"]
        if mid not in truth_of:
            continue
        smiles = str(r["smiles"]).split(";")
        tkey = truth_of[mid]
        tsmiles = smiles_of[mid]

        rank_strict = next((i for i, s in enumerate(smiles, 1)
                            if (Chem.MolToInchiKey(Chem.MolFromSmiles(s)) or "")[:14] == tkey
                            if Chem.MolFromSmiles(s) is not None), 0)
        tk = taut_key(tsmiles)
        rank_taut = next((i for i, s in enumerate(smiles, 1)
                          if taut_key(s) == tk), 0) if tk else 0
        tf = formula(tsmiles)
        rank_formula = next((i for i, s in enumerate(smiles, 1)
                             if formula(s) == tf), 0) if tf else 0

        rows.append({"molecule_id": mid, "strict": rank_strict,
                     "tautomer": rank_taut, "formula": rank_formula})

    res = pd.DataFrame(rows)
    res["rr_strict"] = res["strict"].map(lambda r: 1 / r if 0 < r <= 25 else 0)
    res["rr_taut"] = res["tautomer"].map(lambda r: 1 / r if 0 < r <= 25 else 0)
    res["rr_formula"] = res["formula"].map(lambda r: 1 / r if 0 < r <= 25 else 0)

    print(f"n = {len(res)}")
    print(f"strict  inchikey14 : MRR@25 {res['rr_strict'].mean():.4f}   "
          f"top-1 {(res['strict'] == 1).mean():.3f}")
    print(f"tautomer-canonical: MRR@25 {res['rr_taut'].mean():.4f}   "
          f"top-1 {(res['tautomer'] == 1).mean():.3f}")
    print(f"formula-equality  : MRR@25 {res['rr_formula'].mean():.4f}   "
          f"top-1 {(res['formula'] == 1).mean():.3f}")

    gained = res[(res["rr_strict"] == 0) & (res["rr_taut"] > 0)]
    print(f"\nmolecules rescued by tautomer matching: {len(gained)} "
          f"(+{gained['rr_taut'].sum() / len(res):.3f} MRR)")
    res.to_csv(PROJECT_ROOT / "benchmark" / "rescore_tautomer.csv", index=False)
    print("per-molecule: benchmark/rescore_tautomer.csv")


if __name__ == "__main__":
    main()
