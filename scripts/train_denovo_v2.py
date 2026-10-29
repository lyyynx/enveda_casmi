"""Train the v2 generative spectrum -> SMILES transformer (DeNovoV2Model).

Tutorial architecture scaled up one step (1024 dim, 6+6 layers, ff 4096),
honest-split aware like scripts/train_denovo.py: trains only on train_cut
spectra (held-out structures never seen), excluding ingest_lib ==
'enveda-180' rows. Reuses the shared tokenizer (models/denovo_tokenizer.json)
so the kernel's v1 fingerprint channel and the v2 generative channel decode
with the same vocabulary.

Sized for the 6GB local GPU: batch 24 + grad-accum 4 in bf16.
"""
import argparse
import glob
import math
import os
import re
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from train_denovo import build_tokenizer, gen_metrics, load_training_data, make_batch

from enveda_casmi26.fp_model import DeNovoV2Model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", type=int, default=500_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--enc-layers", type=int, default=6)
    ap.add_argument("--dec-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--ff-dim", type=int, default=4096)
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--max-peaks", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="drop spectra whose SMILES tokenize longer than this")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=2000,
                    help="linear LR warmup steps (post-LN transformer is unstable without it)")
    ap.add_argument("--min-lr-frac", type=float, default=0.1,
                    help="cosine decay floor as a fraction of peak LR")
    ap.add_argument("--spike-guard", type=float, default=5.0,
                    help="skip batches whose loss exceeds this multiple of the running EMA")
    ap.add_argument("--max-spikes", type=int, default=200,
                    help="consecutive skipped batches before restoring the latest snapshot")
    ap.add_argument("--snap-every", type=int, default=5000,
                    help="rolling snapshot interval (model+optimizer, for divergence rollback)")
    ap.add_argument("--keep-snaps", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--diff-channel", action="store_true",
                    help="add pairwise m/z-difference (autocorrelation/power spectrum) input channel")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train_cut.parquet"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models", "denovo_tokenizer.json"))
    ap.add_argument("--rebuild-tokenizer", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "denovo_v2.pt"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    try:
        import rdkit  # noqa: F401

        have_rdkit = True
    except ImportError:
        have_rdkit = False
        print("WARNING: rdkit unavailable - generation metrics disabled; "
              "checkpoints saved per epoch instead of best-by-exact-hit", flush=True)

    t0 = time.time()
    (train_df, train_mz, train_int), (val_df, val_mz, val_int) = load_training_data(
        args.data, args.spectra, args.max_peaks, args.seed)

    if args.rebuild_tokenizer or not os.path.exists(args.tokenizer):
        tok = build_tokenizer(train_df.normalized_smiles.unique().tolist(), args.vocab_size)
        os.makedirs(os.path.dirname(args.tokenizer) or ".", exist_ok=True)
        tok.save(args.tokenizer)
        print(f"tokenizer rebuilt -> {args.tokenizer}")
    else:
        tok = Tokenizer.from_file(args.tokenizer)
        if tok.get_vocab_size() > args.vocab_size:
            raise SystemExit(f"tokenizer vocab {tok.get_vocab_size()} exceeds "
                             f"--vocab-size {args.vocab_size}; retrain with a larger size")
        print(f"tokenizer reused: {args.tokenizer} (vocab {tok.get_vocab_size()})")

    train_tokens = [np.asarray(tok.encode(s).ids, dtype=np.int64)
                    for s in train_df.normalized_smiles.values]
    val_tokens = [np.asarray(tok.encode(s).ids, dtype=np.int64)
                  for s in val_df.normalized_smiles.values]
    train_prec = train_df.precursor_mz.values.astype(np.float32)
    val_prec = val_df.precursor_mz.values.astype(np.float32)
    val_smiles = val_df.normalized_smiles.values
    val_keys = val_df.inchikey14.values

    # Cap decoder sequence length: one huge SMILES in a batch pads the whole
    # batch to its length (make_batch pads to max) and OOMs the decoder.
    # Anything beyond max_new_tokens (~60) is unproducible at inference anyway.
    keep = np.array([i for i, t in enumerate(train_tokens)
                     if len(t) <= args.max_tokens], dtype=np.int64)
    print(f"dropped {len(train_tokens) - len(keep)} train spectra with "
          f"> {args.max_tokens} tokens")
    train_tokens = [train_tokens[i] for i in keep]
    train_mz = [train_mz[i] for i in keep]
    train_int = [train_int[i] for i in keep]
    train_prec = train_prec[keep]

    vkeep = np.array([i for i, t in enumerate(val_tokens)
                      if len(t) <= args.max_tokens], dtype=np.int64)
    val_tokens = [val_tokens[i] for i in vkeep]
    val_mz = [val_mz[i] for i in vkeep]
    val_int = [val_int[i] for i in vkeep]
    val_prec = val_prec[vkeep]
    val_smiles = val_smiles[vkeep]
    val_keys = val_keys[vkeep]
    print(f"data ready at {time.time() - t0:.0f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeNovoV2Model(args.embed_dim, args.enc_layers, args.dec_layers,
                          args.n_heads, args.ff_dim, vocab_size=args.vocab_size,
                          diff_channel=args.diff_channel).to(device)
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

    def lr_lambda(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        p = min(1.0, (s - args.warmup) / max(1, total_steps - args.warmup))
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    for _ in range(base_step):
        sched.step()

    snaps = []
    for p in glob.glob(args.out + ".snap*.pt"):
        m = re.search(r"snap(\d+)\.pt$", p)
        if m:
            snaps.append((int(m.group(1)), p))
    snaps = [p for _, p in sorted(snaps)][-args.keep_snaps:]

    best_metric = -1.0
    step = base_step
    last_path = args.out + ".last.pt"
    ema = None
    spikes = 0
    pending = False
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
            raw = loss.item() * args.grad_accum
            bad = not math.isfinite(raw) or (ema is not None and raw > args.spike_guard * ema)
            if bad:
                spikes += 1
                if spikes % 25 == 0:
                    print(f"  WARNING: {spikes} consecutive loss spikes "
                          f"(loss {raw:.3f}, ema {'n/a' if ema is None else f'{ema:.3f}'})",
                          flush=True)
                if spikes >= args.max_spikes and snaps:
                    ck = torch.load(snaps[-1], map_location=device, weights_only=True)
                    model.load_state_dict(ck["state_dict"])
                    opt.load_state_dict(ck["optimizer"])
                    ema = None
                    spikes = 0
                    pending = False
                    opt.zero_grad(set_to_none=True)
                    print(f"  *** sustained divergence: restored {snaps[-1]} "
                          f"(step {ck.get('step', '?')})", flush=True)
            else:
                spikes = 0
                ema = raw if ema is None else 0.99 * ema + 0.01 * raw
                loss.backward()
                pending = True
            if (g // args.batch_size + 1) % args.grad_accum == 0 or g + args.batch_size >= len(order):
                if not pending:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                pending = False
                step += 1
                if step % 500 == 0:
                    torch.save({"state_dict": model.state_dict(),
                                "optimizer": opt.state_dict(),
                                "config": vars(args), "step": step}, last_path)
                if step % args.snap_every == 0:
                    p = f"{args.out}.snap{step}.pt"
                    torch.save({"state_dict": model.state_dict(),
                                "optimizer": opt.state_dict(),
                                "config": vars(args), "step": step}, p)
                    snaps.append(p)
                    while len(snaps) > args.keep_snaps:
                        old = snaps.pop(0)
                        if os.path.exists(old):
                            os.remove(old)
                if step % 100 == 0:
                    print(f"ep {epoch} step {step}/{total_steps} loss {raw:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
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
                if step % 1500 == 0 and have_rdkit:
                    model.eval()
                    hits, valid, tani = gen_metrics(
                        model, tok, device, val_mz, val_int, val_prec, val_smiles, val_keys)
                    print(f"  GEN: exact {hits:.3f} valid {valid:.3f} tanimoto {tani:.3f}", flush=True)
                    if tani > best_metric:
                        best_metric = tani
                        torch.save({
                            "state_dict": model.state_dict(),
                            "config": vars(args),
                            "val_exact_hit": hits,
                            "val_tanimoto": tani,
                        }, args.out)
                        print(f"  saved -> {args.out} (best tanimoto)", flush=True)
                    model.train()
        if not have_rdkit:
            torch.save({"state_dict": model.state_dict(),
                        "config": vars(args), "step": step}, args.out)
            print(f"epoch {epoch} checkpoint -> {args.out}", flush=True)
    if have_rdkit:
        model.eval()
        hits, valid, tani = gen_metrics(
            model, tok, device, val_mz, val_int, val_prec, val_smiles, val_keys)
        print(f"FINAL GEN: exact {hits:.3f} valid {valid:.3f} tanimoto {tani:.3f}")
        if tani > best_metric:
            best_metric = tani
            torch.save({"state_dict": model.state_dict(), "config": vars(args),
                        "val_exact_hit": hits, "val_tanimoto": tani}, args.out)
            print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
