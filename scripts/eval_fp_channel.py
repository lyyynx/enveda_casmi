"""Honest evaluation of the neural-fingerprint channel on test_cut molecules.

For each held-out molecule: predict a Morgan fingerprint from its spectra
(encoder + fp head), retrieve candidates from the real candidate pool
(mass window over library + external structures), rank by Tanimoto,
and measure MRR@25 vs the v8 retrieval baseline.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_denovo import DeNovoModel, make_batch
from scripts.train_fp_head import morgan_bits
from enveda_casmi26.pipeline import neutral_mass

N_BITS = 2048
TOL_PPM = 8.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="models/denovo_v1.pt")
    ap.add_argument("--head", default="models/fp_head.pt")
    ap.add_argument("--n-mols", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--out", default="benchmark/fp_channel_eval.csv")
    args = ap.parse_args()

    import pandas as pd
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import Chem

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_ckpt = torch.load(args.encoder, map_location=device, weights_only=True)
    cfg = enc_ckpt["config"]
    model = DeNovoModel(cfg["embed_dim"], cfg["n_layers"], cfg["n_heads"],
                        cfg["ff_dim"], vocab_size=cfg["vocab_size"],
                        diff_channel=cfg.get("diff_channel", False))
    model.load_state_dict(enc_ckpt["state_dict"])
    model.eval().to(device)
    head_ckpt = torch.load(args.head, map_location=device, weights_only=True)
    hcfg = head_ckpt["config"]
    head = torch.nn.Sequential(
        torch.nn.Linear(hcfg["in_dim"], hcfg["hidden"]), torch.nn.ReLU(),
        torch.nn.Dropout(0.2), torch.nn.Linear(hcfg["hidden"], hcfg["n_bits"])).to(device)
    head.load_state_dict(head_ckpt["state_dict"])
    head.eval()
    print("encoder + head loaded")

    test = pd.read_parquet(r"data\test_cut.parquet")
    if args.n_mols:
        mol_ids = test.molecule_id.unique()[: args.n_mols]
        test = test[test.molecule_id.isin(mol_ids)]
    slices = test.drop_duplicates("molecule_id").set_index("molecule_id")["slice"] \
        if "slice" in test.columns else None

    print("building candidate pool ...")
    lib = pd.read_parquet(r"data\train_cut.parquet",
                          columns=["inchikey14", "normalized_smiles"])
    lib = lib.dropna(subset=["normalized_smiles"]).drop_duplicates("inchikey14")
    ext = pd.read_parquet(r"data\val_external_pool.parquet")
    pool = pd.DataFrame({
        "inchikey14": np.concatenate([lib.inchikey14.values, ext.inchikey14.values]),
        "smiles": np.concatenate([lib.normalized_smiles.values, ext.smiles.values]),
    })
    lib_mass = np.load("data/lib_mass_cache.npy") if os.path.exists("data/lib_mass_cache.npy") else None
    if lib_mass is None or len(lib_mass) != len(lib):
        from rdkit.Chem import rdMolDescriptors
        lib_mass = np.array([
            rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(s))
            for s in lib.normalized_smiles.values])
        np.save("data/lib_mass_cache.npy", lib_mass)
    pool_mass = np.concatenate([lib_mass, ext.mass.values.astype(float)])
    order = np.argsort(pool_mass)
    pool_mass = pool_mass[order]
    pool_keys = pool.inchikey14.values[order]
    pool_smiles = pool.smiles.values[order]
    print(f"pool: {len(pool)} structures")

    fp_cache: dict[str, np.ndarray] = {}

    def fp_of(smiles):
        if smiles not in fp_cache:
            fp_cache[smiles] = morgan_bits(smiles, fp_gen, N_BITS)
        return fp_cache[smiles]

    rows = []
    t0 = time.time()
    for mi, (mid, g) in enumerate(test.groupby("molecule_id", sort=True)):
        truth_key = g.inchikey14.iloc[0]
        mz_list, int_list, prec_list = [], [], []
        for r in g.itertuples(index=False):
            mzs = np.asarray(r.ms2_mzs, dtype=np.float32)
            ints = np.asarray(r.ms2_normalized_intensities, dtype=np.float32)
            if len(mzs) == 0:
                continue
            order_ = np.argsort(ints)[::-1]
            mzs = mzs[order_][:127]
            ints = ints[order_][:127]
            ints = ints / max(ints.max(), 1e-9)
            mz_list.append(mzs)
            int_list.append(ints)
            prec_list.append(float(r.precursor_mz))
        if not mz_list:
            continue
        nmass = neutral_mass(g.precursor_mz.iloc[0], g.adduct.iloc[0])
        lo = np.searchsorted(pool_mass, nmass * (1 - TOL_PPM / 1e6))
        hi = np.searchsorted(pool_mass, nmass * (1 + TOL_PPM / 1e6))
        cand_keys = pool_keys[lo:hi]
        cand_smiles = pool_smiles[lo:hi]

        probs = []
        with torch.no_grad():
            for s in range(0, len(mz_list), 16):
                mz_b, int_b, mask_b, _ = make_batch(
                    mz_list[s: s + 16], int_list[s: s + 16],
                    np.asarray(prec_list[s: s + 16]), [[1]], 0)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    mem = model.encode(mz_b.to(device), int_b.to(device), mask_b.to(device))
                m = mask_b.to(device).unsqueeze(-1)
                pooled = (mem * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                p = torch.sigmoid(head(pooled.float()))
                probs.append(p.cpu().numpy())
        pred_fp = (np.concatenate(probs).mean(axis=0) > 0.5)
        pred_int = pred_fp.astype(np.int32)

        sims = np.zeros(len(cand_keys))
        for ci, cs in enumerate(cand_smiles):
            cf = fp_of(cs)
            if cf is None:
                continue
            inter = int((pred_int * cf).sum())
            union = int(pred_fp.sum() + cf.sum() - inter)
            sims[ci] = inter / max(1, union)
        best = {}
        for ci in np.argsort(-sims):
            k = cand_keys[ci]
            if k not in best:
                best[k] = sims[ci]
            if len(best) >= args.top_k:
                break
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        keys = [k for k, _ in ranked]
        any_hit = truth_key in keys
        rank = keys.index(truth_key) + 1 if any_hit else 0
        rows.append({"molecule_id": mid, "truth_key": truth_key,
                     "n_cands": len(cand_keys), "fp_top1": int(keys[0] == truth_key) if keys else 0,
                     "hit": any_hit, "rank": rank,
                     "rr": 1.0 / rank if any_hit else 0.0})
        if (mi + 1) % 50 == 0:
            print(f"  {mi + 1} molecules ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nFP-ONLY: MRR@25={out.rr.mean():.4f} top-1={out.fp_top1.mean():.4f} "
          f"any={out['hit'].mean():.4f} n={len(out)}")
    if slices is not None:
        out2 = out.merge(slices, left_on="molecule_id", right_index=True, how="left")
        for sl, grp in out2.groupby("slice"):
            print(f"  {sl}: n={len(grp)} MRR@25={grp.rr.mean():.4f} top-1={grp.fp_top1.mean():.4f}")


if __name__ == "__main__":
    main()
