"""Train a Morgan-fingerprint head on the frozen de novo transformer encoder.

Spectrum -> encoder (frozen, from models/denovo_v1.pt) -> masked-mean pooling
-> MLP head -> 2048-bit Morgan radius-2 fingerprint (BCE).

Honest-split aware: same 500-structure validation holdout as the transformer
training; test_cut molecules are never touched here.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_denovo import DeNovoModel, make_batch


def morgan_bits(smiles, fp_gen, n_bits):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = fp_gen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=bool)
    for b in fp.GetOnBits():
        arr[b] = True
    return arr


def embed_spectra(model, device, mz_list, int_list, prec_list, batch=256):
    outs = []
    with torch.no_grad():
        for g in range(0, len(mz_list), batch):
            mz_b, int_b, mask_b, _ = make_batch(
                mz_list[g: g + batch], int_list[g: g + batch],
                prec_list[g: g + batch], [[1]], 0)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                mem = model.encode(mz_b.to(device), int_b.to(device), mask_b.to(device))
            m = mask_b.to(device).unsqueeze(-1)
            pooled = (mem * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            outs.append(pooled.float().cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="models/denovo_v1.pt")
    ap.add_argument("--cache", default="data/fp_embed_cache.npz")
    ap.add_argument("--out", default="models/fp_head.pt")
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    import pandas as pd
    from rdkit.Chem import rdFingerprintGenerator

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=args.n_bits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.encoder, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = DeNovoModel(cfg["embed_dim"], cfg["n_layers"], cfg["n_heads"],
                        cfg["ff_dim"], vocab_size=cfg["vocab_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"encoder loaded (val_exact_hit={ckpt.get('val_exact_hit')})")

    if os.path.exists(args.cache):
        print("loading embedding cache ...")
        cache = np.load(args.cache, allow_pickle=True)
        x_all = cache["x"]
        y_all = np.unpackbits(cache["y"], axis=1).astype(bool)
        keys_all = cache["keys"]
        is_val = cache["is_val"]
        meta = pd.read_parquet(r"data\train_cut.parquet",
                               columns=["inchikey14", "normalized_smiles"])
        meta = meta.dropna(subset=["normalized_smiles"]).drop_duplicates("inchikey14")
        smiles_map = dict(zip(meta.inchikey14, meta.normalized_smiles))
        spec_keys = [str(k) for k in keys_all[is_val]]
        val_keys = sorted(set(spec_keys))
        val_smiles = [smiles_map[k] for k in val_keys]
    else:
        print("reading data ...")
        df = pd.read_parquet(r"data\train_cut.parquet", columns=[
            "inchikey14", "normalized_smiles", "precursor_mz",
            "ms2_mzs", "ms2_normalized_intensities", "ingest_lib"])
        df = df[df.ingest_lib != "enveda-180"].dropna(subset=["normalized_smiles"])
        rng = np.random.RandomState(args.seed)
        structs = df.inchikey14.unique()
        val_structs = set(structs[rng.permutation(len(structs))[:500]])
        is_val = df.inchikey14.isin(val_structs).values
        val_df = df[is_val].drop_duplicates(subset=["inchikey14"], keep="first")
        train_df = df[~is_val]

        print(f"computing {len(df)} spectra embeddings ({len(train_df)} train / {len(val_df)} val) ...")
        t0 = time.time()
        mz_list, int_list, prec_list = [], [], []
        keep = np.ones(len(df), dtype=bool)
        for i, r in enumerate(df.itertuples(index=False)):
            mzs = np.asarray(r.ms2_mzs, dtype=np.float32)
            ints = np.asarray(r.ms2_normalized_intensities, dtype=np.float32)
            if len(mzs) == 0:
                keep[i] = False
                mz_list.append(np.zeros(0, dtype=np.float32))
                int_list.append(np.zeros(0, dtype=np.float32))
                continue
            order = np.argsort(ints)[::-1]
            mzs = mzs[order][:127]
            ints = ints[order][:127]
            ints = ints / max(ints.max(), 1e-9)
            mz_list.append(mzs)
            int_list.append(ints)
        df_reset = df[keep]
        prec_list = df_reset.precursor_mz.values.astype(np.float32)
        mz_list = [m for m, k in zip(mz_list, keep) if k]
        int_list = [m for m, k in zip(int_list, keep) if k]
        is_val = is_val[keep]

        x_all = embed_spectra(model, device, mz_list, int_list, prec_list)
        print(f"embedded in {time.time() - t0:.0f}s")

        print("computing fingerprint targets ...")
        fps = {}
        y_rows = []
        for s in df_reset.normalized_smiles.values:
            if s not in fps:
                fps[s] = morgan_bits(s, fp_gen, args.n_bits)
            y_rows.append(fps[s])
        y_all = np.stack(y_rows).astype(np.float32)
        keys_all = df_reset.inchikey14.values
        val_df = df_reset[is_val].drop_duplicates(subset=["inchikey14"], keep="first")
        val_idx = df_reset.index[is_val].get_indexer(val_df.index)
        val_smiles = df_reset.normalized_smiles.values[val_idx].tolist()
        val_keys = df_reset.inchikey14.values[val_idx].tolist()
        np.savez_compressed(
            args.cache, x=x_all.astype(np.float16),
            y=np.packbits(y_all.astype(bool), axis=1),
            keys=keys_all, is_val=is_val,
            val_smiles=np.array(val_smiles, dtype=object),
            val_keys=np.array(val_keys, dtype=object))
        print(f"cache saved -> {args.cache}")

    x_val = x_all[is_val]
    y_val = y_all[is_val]
    x_train = x_all[~is_val]
    y_train = y_all[~is_val]
    print(f"train: {len(x_train)} spectra / {len(set(keys_all[~is_val]))} structures")
    print(f"val:   {len(x_val)} spectra ({len(val_keys)} structures)")

    head = nn.Sequential(
        nn.Linear(x_train.shape[1], args.hidden), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(args.hidden, args.n_bits)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pos_weight = torch.tensor((neg / np.maximum(pos, 1)).clip(1, 50),
                              dtype=torch.float32, device=device)

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train)
    xv = torch.from_numpy(x_val).to(torch.float32).to(device)
    yv_bits = y_val.astype(bool)

    val_fps_list = []
    for s in val_smiles:
        fp = morgan_bits(s, fp_gen, args.n_bits)
        val_fps_list.append(fp if fp is not None else np.zeros(args.n_bits, dtype=bool))
    val_fps_truth = np.stack(val_fps_list)
    val_keys_arr = np.array(val_keys)
    key_to_idx = {k: i for i, k in enumerate(val_keys)}
    spec_keys = keys_all[is_val]
    own_idx = np.array([key_to_idx[k] for k in spec_keys])
    truth_mat = torch.from_numpy(val_fps_truth.astype(np.float32)).to(device)

    best = -1.0
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(len(xt))
        losses = []
        for g in range(0, len(xt), args.batch_size):
            idx = perm[g: g + args.batch_size]
            xb = xt[idx].to(torch.float32).to(device)
            yb = yt[idx].to(torch.float32).to(device)
            logits = head(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (g // args.batch_size) % 50 == 0:
                print(f"  ep {epoch} batch {g // args.batch_size}/"
                      f"{len(xt) // args.batch_size} loss {np.mean(losses[-50:]):.4f}",
                      flush=True)
        head.eval()
        with torch.no_grad():
            pv = (head(xv) > 0).cpu().numpy()
        tan = np.array([
            np.logical_and(pv[i], yv_bits[i]).sum() /
            max(1, np.logical_or(pv[i], yv_bits[i]).sum())
            for i in range(len(pv))])
        mean_tan = tan.mean()
        pred = torch.from_numpy(pv.astype(np.float32)).to(device)
        sim = (pred @ truth_mat.T) / (
            (pred.sum(1, keepdim=True) + truth_mat.sum(1).unsqueeze(0)
             - pred @ truth_mat.T)).clamp(min=1)
        order = sim.argsort(dim=1, descending=True).cpu().numpy()
        top1 = np.mean([o[0] == w for o, w in zip(order, own_idx)])
        top25 = np.mean([w in o[:25] for o, w in zip(order, own_idx)])
        print(f"ep {epoch}: loss {np.mean(losses):.4f} tanimoto {mean_tan:.3f} "
              f"retrieval top1 {top1:.3f} top25 {top25:.3f}", flush=True)
        if top25 > best:
            best = top25
            torch.save({"state_dict": head.state_dict(),
                       "config": {"in_dim": x_train.shape[1], "hidden": args.hidden,
                                  "n_bits": args.n_bits, "pooling": "masked_mean",
                                  "encoder_ckpt": args.encoder}}, args.out)
            print(f"  saved -> {args.out}")
    print(f"done. best val top25 {best:.3f}")


if __name__ == "__main__":
    main()
