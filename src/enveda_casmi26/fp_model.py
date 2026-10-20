"""De novo transformer encoder + fingerprint head (single source of truth).

Mirrored into the Kaggle dataset as code_fp.py; the kernel and all local
training/eval scripts share these definitions.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PeakEmbedder(nn.Module):
    def __init__(self, d_model, dropout, mz_log_lims=(-2.0, 3.0)):
        super().__init__()
        sin_dim = d_model
        wavelength = torch.pow(
            10,
            (mz_log_lims[1] - mz_log_lims[0])
            * torch.linspace(0, 1, sin_dim // 2)
            + mz_log_lims[0],
        )
        self._frequency = nn.Parameter(2 * np.pi / wavelength, requires_grad=False)
        self.ff1 = nn.Sequential(
            nn.Linear(sin_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.Dropout(dropout),
        )
        self.ff2 = nn.Sequential(
            nn.Linear(d_model + 1, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, mz, inten):
        omega = self._frequency.view(1, 1, -1) * mz.unsqueeze(-1)
        mz_embeds = self.ff1(torch.cat([torch.sin(omega), torch.cos(omega)], dim=-1))
        return self.ff2(torch.cat([mz_embeds, inten.unsqueeze(-1)], dim=-1))


class DeNovoModel(nn.Module):
    def __init__(self, embed_dim=512, n_layers=4, n_heads=8, ff_dim=2048,
                 dropout=0.1, vocab_size=512, pad_id=0, bos_id=1, eos_id=2):
        super().__init__()
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.peak_embedder = PeakEmbedder(embed_dim, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            embed_dim, n_heads, ff_dim, dropout, "gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.wte = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        dec_layer = nn.TransformerDecoderLayer(
            embed_dim, n_heads, ff_dim, dropout, "gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        nn.init.normal_(self.wte.weight, std=1.0)
        for mod in list(self.encoder.modules()) + list(self.decoder.modules()):
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)
            elif isinstance(mod, nn.LayerNorm):
                nn.init.constant_(mod.weight, 1.0)
                nn.init.constant_(mod.bias, 0.0)

    def encode(self, mz, inten, mask):
        x = self.peak_embedder(mz, inten)
        return self.encoder(x, src_key_padding_mask=(mask == 0))

    def pooled_encoding(self, mz, inten, mask):
        mem = self.encode(mz, inten, mask)
        m = mask.unsqueeze(-1)
        return (mem * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward(self, tokens, memory, enc_mask):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tokens.shape[1], tokens.device)
        x = self.decoder(
            tgt=self.wte(tokens), memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=(enc_mask == 0),
        )
        logits = 15 * torch.tanh(self.lm_head(x) / 15)
        return logits.float()

    def loss(self, tokens, memory, enc_mask):
        logits = self.forward(tokens[:, :-1], memory, enc_mask)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1),
            ignore_index=self.pad_id)

    @torch.inference_mode()
    def generate(self, memory, enc_mask, n_samples=5, max_new_tokens=60, temperature=1.0):
        b, s, d = memory.shape
        idx = torch.full((b * n_samples, 1), self.bos_id, dtype=torch.long, device=memory.device)
        logprob = torch.zeros(b * n_samples, device=memory.device)
        mem = memory.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(b * n_samples, s, d)
        msk = enc_mask.unsqueeze(1).expand(-1, n_samples, -1).reshape(b * n_samples, s)
        done = torch.zeros(b * n_samples, dtype=torch.bool, device=memory.device)
        for _ in range(max_new_tokens):
            logits = self.forward(idx, mem, msk)[:, -1]
            lp = logits.log_softmax(dim=-1)
            if done.any():
                lp[done] = -float("inf")
                lp[done, self.eos_id] = 0.0
            probs = (lp / temperature).softmax(dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
            logprob = logprob + torch.where(done, torch.zeros_like(logprob), lp.gather(1, nxt.unsqueeze(1)).squeeze(1))
            idx = torch.cat([idx, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == self.eos_id)
            if done.all():
                break
        return idx.reshape(b, n_samples, -1), logprob.reshape(b, n_samples)


def build_fp_head(cfg: dict) -> nn.Module:
    """Head architecture must match scripts/train_fp_head.py exactly."""
    return nn.Sequential(
        nn.Linear(cfg["in_dim"], cfg["hidden"]), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(cfg["hidden"], cfg["n_bits"]))


def load_fp_model(encoder_path, head_path, device=None):
    """Load the fp encoder + head for inference (weights fp32 on any device)."""
    device = device or torch.device("cpu")
    enc = torch.load(encoder_path, map_location=device, weights_only=True)
    cfg = enc["config"]
    model = DeNovoModel(cfg["embed_dim"], cfg["n_layers"], cfg["n_heads"],
                        cfg["ff_dim"], vocab_size=cfg["vocab_size"])
    model.load_state_dict(enc["state_dict"])
    model.eval().to(device)
    head = None
    if head_path is not None:
        h = torch.load(head_path, map_location=device, weights_only=True)
        head = build_fp_head(h["config"])
        head.load_state_dict(h["state_dict"])
        head.eval().to(device)
    return model, head


def prep_spectrum(mzs, intensities, max_peaks: int = 127):
    mzs = np.asarray(mzs, dtype=np.float32)
    ints = np.asarray(intensities, dtype=np.float32)
    if len(mzs) == 0:
        return mzs, ints
    order = np.argsort(ints)[::-1]
    mzs = mzs[order][:max_peaks]
    ints = ints[order][:max_peaks]
    ints = ints / max(float(ints.max()), 1e-9)
    return mzs, ints


def encode_batch(model, mz_list, int_list, prec_list, device, batch: int = 64):
    """Masked-mean pooled encoder embeddings for a list of spectra."""
    out = []
    with torch.no_grad():
        for g in range(0, len(mz_list), batch):
            peaks = max(len(m) for m in mz_list[g: g + batch]) + 1
            b = len(mz_list[g: g + batch])
            mzs = np.zeros((b, peaks), dtype=np.float32)
            ints = np.zeros((b, peaks), dtype=np.float32)
            mask = np.zeros((b, peaks), dtype=np.float32)
            for i, (m, inten) in enumerate(zip(mz_list[g: g + batch], int_list[g: g + batch])):
                k = len(m) + 1
                mzs[i, 0] = prec_list[g + i]
                mzs[i, 1:k] = m
                ints[i, 0] = 2.0
                ints[i, 1:k] = inten
                mask[i, :k] = 1.0
            mzs_t = torch.from_numpy(mzs).to(device)
            ints_t = torch.from_numpy(ints).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            amp = device.type == "cuda"
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pooled = model.pooled_encoding(mzs_t, ints_t, mask_t)
            else:
                pooled = model.pooled_encoding(mzs_t, ints_t, mask_t)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.peak_embedder.ff1[0].out_features))


def predict_molecule_fp(model, head, spectra, device) -> np.ndarray | None:
    """Mean predicted fingerprint bits over a molecule's spectra (bool array)."""
    mz_list, int_list, prec_list = [], [], []
    for spec in spectra:
        mzs, ints = prep_spectrum(spec[0], spec[1])
        if len(mzs) == 0:
            continue
        mz_list.append(mzs)
        int_list.append(ints)
        prec_list.append(float(spec[2]))
    if not mz_list:
        return None
    pooled = encode_batch(model, mz_list, int_list, prec_list, device)
    t = torch.from_numpy(pooled).to(next(head.parameters()).device)
    probs = torch.sigmoid(head(t)).cpu().numpy()
    return (probs.mean(axis=0) > 0.5).astype(bool)
