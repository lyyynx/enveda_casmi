"""Evaluate the de novo transformer honestly on the held-out test_cut molecules.

Measures: SMILES validity, exact ikey14 top-1 / any-hit among generated candidates,
and how generation would complement the v8 retrieval channel (rescue analysis).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_denovo import (DeNovoModel, make_batch, decode_ids, canonical)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = DeNovoModel(cfg["embed_dim"], cfg["n_layers"], cfg["n_heads"],
                        cfg["ff_dim"], vocab_size=cfg["vocab_size"],
                        diff_channel=cfg.get("diff_channel", False))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/denovo_v1.pt")
    ap.add_argument("--tokenizer", default="models/denovo_tokenizer.json")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--max-spectra", type=int, default=4)
    ap.add_argument("--n-mols", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--retrieval", default="benchmark/validate_results.csv")
    ap.add_argument("--out", default="benchmark/denovo_eval.csv")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    import pandas as pd
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    tok = Tokenizer.from_file(args.tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, device)
    print(f"ckpt val_exact_hit={ckpt.get('val_exact_hit')} on {device}")

    df = pd.read_parquet(r"data\test_cut.parquet")
    mol_ids = df.molecule_id.unique()
    if args.n_mols:
        mol_ids = mol_ids[: args.n_mols]
    print(f"{len(mol_ids)} molecules, {len(df)} spectra")

    rows = []
    t0 = time.time()
    for mi, mid in enumerate(mol_ids):
        g = df[df.molecule_id == mid].iloc[: args.max_spectra]
        truth_key = g.inchikey14.iloc[0]
        cands = {}
        mz_list, int_list, prec_list = [], [], []
        for r in g.itertuples(index=False):
            mzs = np.asarray(r.ms2_mzs, dtype=np.float32)
            ints = np.asarray(r.ms2_normalized_intensities, dtype=np.float32)
            if len(mzs) == 0:
                continue
            order = np.argsort(ints)[::-1]
            mzs = mzs[order][:127]
            ints = ints[order][:127]
            ints = ints / max(ints.max(), 1e-9)
            mz_list.append(mzs)
            int_list.append(ints)
            prec_list.append(float(r.precursor_mz))
        n_spec = len(mz_list)
        if n_spec == 0:
            rows.append({"molecule_id": mid, "truth_key": truth_key, "n_spectra": 0,
                         "n_generated": 0, "n_valid": 0, "top1_hit": 0, "any_hit": 0,
                         "truth_rank": None, "best_tanimoto": None})
            continue
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            mz_b, int_b, mask_b, _ = make_batch(mz_list, int_list, np.asarray(prec_list), [[1]], 0)
            memory = model.encode(mz_b.to(device), int_b.to(device), mask_b.to(device))
            gen, scores = model.generate(memory, mask_b.to(device),
                                         n_samples=args.samples,
                                         temperature=args.temperature)
        n_valid = 0
        for s in range(n_spec):
            for j in range(args.samples):
                ids = gen[s, j].tolist()
                if 2 in ids:
                    ids = ids[: ids.index(2)]
                smiles = decode_ids([t for t in ids if t not in (0, 1, 2)], tok)
                c, key = canonical(smiles)
                if c is None:
                    continue
                n_valid += 1
                lp = float(scores[s, j])
                if key not in cands or lp > cands[key][0]:
                    cands[key] = (lp, c)
        ranked = sorted(cands.items(), key=lambda kv: -kv[1][0])
        keys = [k for k, _ in ranked]
        top1 = int(keys[0] == truth_key) if keys else 0
        any_hit = int(truth_key in keys)
        rank = keys.index(truth_key) + 1 if any_hit else None
        rows.append({"molecule_id": mid, "truth_key": truth_key, "n_spectra": n_spec,
                     "n_generated": args.samples * n_spec, "n_valid": n_valid,
                     "top1_hit": top1, "any_hit": any_hit, "truth_rank": rank,
                     "n_unique_valid": len(keys)})
        if (mi + 1) % 50 == 0:
            print(f"  {mi + 1}/{len(mol_ids)} molecules ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs("benchmark", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nvalidity: {out.n_valid.sum() / max(1, out.n_generated.sum()):.3f}")
    print(f"top-1 exact: {out.top1_hit.mean():.3f}")
    print(f"any-hit (<= {args.samples}x{args.max_spectra} samples): {out.any_hit.mean():.3f}")

    if os.path.exists(args.retrieval):
        ret = pd.read_csv(args.retrieval)
        if "hit" in ret.columns:
            m = out.merge(ret[["molecule_id", "hit"]], on="molecule_id", how="left")
            m = m.dropna(subset=["hit"])
            rescued = m[(m.hit == 0) & (m.any_hit == 1)]
            print(f"rescue (retrieval missed, de novo hit): "
                  f"{len(rescued)}/{len(m)} = {len(rescued) / max(1, len(m)):.3f}")
            both = m[(m.hit == 1) & (m.any_hit == 1)]
            print(f"both channels hit: {len(both)}/{len(m)}")
            m.to_csv(args.out.replace(".csv", "_joined.csv"), index=False)


if __name__ == "__main__":
    main()
