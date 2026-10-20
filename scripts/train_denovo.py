"""Train the de novo spectrum -> SMILES transformer (tutorial architecture, scaled to 6GB GPU).

Honest-split aware: trains only on train_cut spectra (held-out structures never seen),
excluding ingest_lib == 'enveda-180' rows like the organizers' tutorial.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, processors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from enveda_casmi26.fp_model import DeNovoModel

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- data
def process_spectrum(mzs, intensities, max_len):
    mzs = np.asarray(mzs, dtype=np.float32)
    ints = np.asarray(intensities, dtype=np.float32)
    if len(mzs) == 0:
        return mzs, ints
    order = np.argsort(ints)[::-1]
    mzs = mzs[order][: max_len - 1]
    ints = ints[order][: max_len - 1]
    if ints.max() > 0:
        ints = ints / ints.max()
    return mzs, ints


def load_training_data(path, n_spectra, max_len, seed=0):
    import pandas as pd

    cols = ["inchikey14", "normalized_smiles", "precursor_mz", "ms2_mzs",
            "ms2_normalized_intensities", "ingest_lib"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df.ingest_lib != "enveda-180"]
    df = df.dropna(subset=["normalized_smiles"])
    print(f"after enveda-180 filter: {len(df)} spectra")

    rng = np.random.RandomState(seed)
    structs = df.inchikey14.unique()
    val_structs = set(structs[rng.permutation(len(structs))[:500]])
    is_val = df.inchikey14.isin(val_structs)
    val_df = df[is_val].drop_duplicates(subset=["inchikey14"], keep="first")
    train_df = df[~is_val]

    if n_spectra and len(train_df) > n_spectra:
        train_df = train_df.sample(n=n_spectra, random_state=seed)
    print(f"train: {len(train_df)} spectra / {train_df.inchikey14.nunique()} structures")
    print(f"val:   {len(val_df)} spectra (one per structure)")

    def prep(d):
        mz_list, int_list = [], []
        for mzs, ints in zip(d.ms2_mzs.values, d.ms2_normalized_intensities.values):
            m, i = process_spectrum(mzs, ints, max_len)
            mz_list.append(m)
            int_list.append(i)
        return mz_list, int_list

    train_mz, train_int = prep(train_df)
    val_mz, val_int = prep(val_df)
    return (train_df, train_mz, train_int), (val_df, val_mz, val_int)


def build_tokenizer(smiles_list, vocab_size):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    alphabet = sorted(set("".join(smiles_list)))
    tok = Tokenizer(models.BPE())
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<pad>", 0), ("<s>", 1), ("</s>", 2)],
    )
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=alphabet,
        special_tokens=["<pad>", "<s>", "</s>"],
        show_progress=False,
    )
    tok.train_from_iterator(iter(smiles_list), trainer)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return tok


# ---------------------------------------------------------------- batching
def make_batch(mz_list, int_list, precursors, token_list, pad_id):
    peaks = max(len(m) for m in mz_list) + 1
    b = len(mz_list)
    mzs = np.zeros((b, peaks), dtype=np.float32)
    ints = np.zeros((b, peaks), dtype=np.float32)
    mask = np.zeros((b, peaks), dtype=np.float32)
    for i, (m, inten) in enumerate(zip(mz_list, int_list)):
        k = len(m) + 1
        mzs[i, 0] = precursors[i]
        mzs[i, 1:k] = m
        ints[i, 0] = 2.0
        ints[i, 1:k] = inten
        mask[i, :k] = 1.0
    tlen = max(len(t) for t in token_list)
    toks = np.full((b, tlen), pad_id, dtype=np.int64)
    for i, t in enumerate(token_list):
        toks[i, : len(t)] = t
    return (torch.from_numpy(mzs), torch.from_numpy(ints),
            torch.from_numpy(mask), torch.from_numpy(toks))


def decode_ids(ids, tokenizer):
    return tokenizer.decode(ids).replace(" ", "")


def canonical(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol).split("-")[0]


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", type=int, default=500_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ff-dim", type=int, default=2048)
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--max-peaks", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--out", default="models/denovo_v1.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    t0 = time.time()
    (train_df, train_mz, train_int), (val_df, val_mz, val_int) = load_training_data(
        r"data\train_cut.parquet", args.spectra, args.max_peaks, args.seed)

    tok = build_tokenizer(train_df.normalized_smiles.unique().tolist(), args.vocab_size)
    os.makedirs("models", exist_ok=True)
    tok.save("models/denovo_tokenizer.json")

    train_tokens = [np.asarray(tok.encode(s).ids, dtype=np.int64)
                    for s in train_df.normalized_smiles.values]
    val_tokens = [np.asarray(tok.encode(s).ids, dtype=np.int64)
                  for s in val_df.normalized_smiles.values]
    train_prec = train_df.precursor_mz.values.astype(np.float32)
    val_prec = val_df.precursor_mz.values.astype(np.float32)
    val_smiles = val_df.normalized_smiles.values
    val_keys = val_df.inchikey14.values
    print(f"data ready at {time.time() - t0:.0f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeNovoModel(args.embed_dim, args.n_layers, args.n_heads, args.ff_dim,
                        vocab_size=args.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    base_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        base_step = ckpt.get("step", 0)
        print(f"resumed from {args.resume} at step {base_step} "
              f"(val_exact_hit {ckpt.get('val_exact_hit', '?')})")
    scaler_dtype = torch.bfloat16

    steps_per_epoch = max(1, len(train_mz) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch} optim steps/epoch")
    best_metric = -1.0
    step = base_step
    last_path = args.out + ".last.pt"
    for epoch in range(args.epochs):
        order = np.random.RandomState(args.seed + epoch + 1000 * base_step).permutation(len(train_mz))
        for g in range(0, len(order), args.batch_size):
            batch_idx = order[g: g + args.batch_size]
            mz_b, int_b, mask_b, tok_b = make_batch(
                [train_mz[i] for i in batch_idx], [train_int[i] for i in batch_idx],
                train_prec[batch_idx], [train_tokens[i] for i in batch_idx], 0)
            with torch.autocast(device_type=device.type, dtype=scaler_dtype):
                memory = model.encode(mz_b.to(device), int_b.to(device), mask_b.to(device))
                loss = model.loss(tok_b.to(device), memory, mask_b.to(device)) / args.grad_accum
            loss.backward()
            if (g // args.batch_size + 1) % args.grad_accum == 0 or g + args.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 500 == 0:
                    torch.save({"state_dict": model.state_dict(),
                                "optimizer": opt.state_dict(),
                                "config": vars(args), "step": step}, last_path)
                if step % 100 == 0:
                    print(f"ep {epoch} step {step}/{total_steps} "
                          f"loss {loss.item() * args.grad_accum:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                if step % 500 == 0:
                    model.eval()
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=scaler_dtype):
                        vm, vi, vk, vt = make_batch(val_mz[:64], val_int[:64], val_prec[:64],
                                                    val_tokens[:64], 0)
                        memory = model.encode(vm.to(device), vi.to(device), vk.to(device))
                        vloss = model.loss(vt.to(device), memory, vk.to(device)).item()
                    print(f"  val loss {vloss:.4f}", flush=True)
                    model.train()
                if step % 1500 == 0:
                    model.eval()
                    hits, valid, tani = gen_metrics(
                        model, tok, device, val_mz, val_int, val_prec, val_smiles, val_keys)
                    print(f"  GEN: exact {hits:.3f} valid {valid:.3f} tanimoto {tani:.3f}", flush=True)
                    if hits > best_metric:
                        best_metric = hits
                        torch.save({
                            "state_dict": model.state_dict(),
                            "config": vars(args),
                            "val_exact_hit": hits,
                        }, args.out)
                        print(f"  saved -> {args.out}", flush=True)
                    model.train()
    model.eval()
    hits, valid, tani = gen_metrics(
        model, tok, device, val_mz, val_int, val_prec, val_smiles, val_keys)
    print(f"FINAL GEN: exact {hits:.3f} valid {valid:.3f} tanimoto {tani:.3f}")
    if hits > best_metric:
        torch.save({"state_dict": model.state_dict(), "config": vars(args),
                    "val_exact_hit": hits}, args.out)
        print(f"saved -> {args.out}")


def gen_metrics(model, tok, device, val_mz, val_int, val_prec, val_smiles, val_keys, n=64):
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    exact, valid_cnt, attempt_cnt, tani_sum = 0, 0, 0, 0.0
    for i in range(min(n, len(val_mz))):
        mz_b, int_b, mask_b, _ = make_batch([val_mz[i]], [val_int[i]], [val_prec[i]], [[1]], 0)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            memory = model.encode(mz_b.to(device), int_b.to(device), mask_b.to(device))
            gen, scores = model.generate(memory, mask_b.to(device), n_samples=8)
        truth_smiles = val_smiles[i]
        truth_key = val_keys[i]
        mol_t = __import__("rdkit").Chem.MolFromSmiles(truth_smiles)
        fp_t = fp_gen.GetSparseCountFingerprint(mol_t) if mol_t else None
        best_tani = 0.0
        seen = set()
        for j in range(gen.shape[1]):
            ids = gen[0, j].tolist()
            if 2 in ids:
                ids = ids[: ids.index(2)]
            attempt_cnt += 1
            smiles = decode_ids([t for t in ids if t not in (0, 1, 2)], tok)
            c, key = canonical(smiles)
            if c is None or key in seen:
                continue
            seen.add(key)
            valid_cnt += 1
            if key == truth_key:
                exact += 1
                break
            if fp_t is not None:
                fp_c = fp_gen.GetSparseCountFingerprint(__import__("rdkit").Chem.MolFromSmiles(c))
                t = DataStructs.TanimotoSimilarity(fp_t, fp_c)
                best_tani = max(best_tani, t)
        tani_sum += best_tani
    n = min(n, len(val_mz))
    return exact / n, valid_cnt / max(1, attempt_cnt), tani_sum / n


if __name__ == "__main__":
    main()
