"""Inspect the COCONUT CASMI26 candidate dataset."""
import pickle
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

meta = pickle.load(open(r"data\coconut\coco_meta.pkl", "rb"))
print("meta keys:", list(meta.keys()))
keys = np.asarray(meta["keys"], dtype=object)
smis = np.asarray(meta["smiles"], dtype=object)
mass = np.load(r"data\coconut\coco_mass.npy")
bits = np.load(r"data\coconut\fp_bits.npy")
print(f"structures : {len(keys):,}")
print(f"mass range : {mass.min():.2f} .. {mass.max():.2f}")
print(f"fp_bits    : {len(bits):,} of {bits.max() + 1} possible positions")
print("example keys :", keys[:3])
print("example smiles:", smis[:3])

from rdkit import Chem  # noqa: E402

ik = Chem.MolToInchiKey(Chem.MolFromSmiles(str(smis[0])))
print(f"example inchikey from smiles: {ik}")
uniq = len(set(keys))
print(f"unique keys: {uniq:,}  (dupes: {len(keys) - uniq:,})")
