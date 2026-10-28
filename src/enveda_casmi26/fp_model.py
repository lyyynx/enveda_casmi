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


class DifferenceChannel(nn.Module):
    """Intensity-weighted histogram of pairwise peak m/z differences.

    The lag histogram of a signal is its autocorrelation, whose Fourier
    transform is the power spectrum (Wiener-Khinchin), so this is the exact
    (non-uniform m/z) form of the spectrum's power spectrum: neutral losses,
    isotope spacing (1.0033 Da) and repeated substructure spacings appear as
    sharp lag peaks. The histogram is projected to d_model and added to
    every encoder memory position as a global per-spectrum bias; adding the
    same vector to all keys leaves attention scores unchanged and the bias
    reaches the decoder through the cross-attention values, so no memory
    length or padding-mask changes are needed downstream.
    """

    def __init__(self, d_model, max_lag=200.0, res=0.05, dropout=0.1):
        super().__init__()
        self.max_lag = max_lag
        self.res = res
        self.n_lags = int(round(max_lag / res)) + 1
        self.proj = nn.Sequential(
            nn.Linear(self.n_lags, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model))

    def forward(self, mz, inten, mask):
        m = mask > 0
        w = inten * m
        d = mz.unsqueeze(2) - mz.unsqueeze(1)
        keep = (d >= 0) & (d < self.max_lag) & m.unsqueeze(2) & m.unsqueeze(1)
        idx = (d * (1.0 / self.res)).long()
        idx = torch.where(keep, idx, torch.zeros_like(idx)).clamp_(max=self.n_lags - 1)
        b = mz.size(0)
        flat = torch.zeros(b * self.n_lags, device=mz.device, dtype=inten.dtype)
        offsets = (torch.arange(b, device=mz.device).view(b, 1, 1)
                   * self.n_lags).expand_as(idx)
        flat.scatter_add_(0, (idx + offsets).reshape(-1),
                          (w.unsqueeze(2) * w.unsqueeze(1)).reshape(-1))
        feats = flat.view(b, self.n_lags)
        return self.proj(feats)


class DeNovoModel(nn.Module):
    def __init__(self, embed_dim=512, n_layers=4, n_heads=8, ff_dim=2048,
                 dropout=0.1, vocab_size=512, pad_id=0, bos_id=1, eos_id=2,
                 diff_channel=False):
        super().__init__()
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.peak_embedder = PeakEmbedder(embed_dim, dropout)
        self.diff_channel = diff_channel
        if diff_channel:
            self.diff = DifferenceChannel(embed_dim, dropout=dropout)
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
        mem = self.encoder(x, src_key_padding_mask=(mask == 0))
        if self.diff_channel:
            mem = mem + self.diff(mz, inten, mask).unsqueeze(1)
        return mem

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


class DeNovoV2Model(DeNovoModel):
    """Generative spectrum -> SMILES transformer, second generation.

    Same training/generation interface as DeNovoModel but a step larger than
    the tutorial reference (768 dim, 6+6 layers, ff 3072) and with separate
    encoder/decoder depths. Used as the third retrieval channel: sampled
    structures are matched against the candidate pool by exact inchikey14
    or fingerprint Tanimoto (see pipeline.ChannelEngine.denovo_scores).
    """

    def __init__(self, embed_dim=1024, enc_layers=6, dec_layers=6, n_heads=16,
                 ff_dim=4096, dropout=0.1, vocab_size=512,
                 pad_id=0, bos_id=1, eos_id=2, diff_channel=False):
        nn.Module.__init__(self)
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.peak_embedder = PeakEmbedder(embed_dim, dropout)
        self.diff_channel = diff_channel
        if diff_channel:
            self.diff = DifferenceChannel(embed_dim, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            embed_dim, n_heads, ff_dim, dropout, "gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)
        self.wte = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        dec_layer = nn.TransformerDecoderLayer(
            embed_dim, n_heads, ff_dim, dropout, "gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
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
                        cfg["ff_dim"], vocab_size=cfg["vocab_size"],
                        diff_channel=cfg.get("diff_channel", False))
    model.load_state_dict(enc["state_dict"])
    model.eval().to(device)
    head = None
    if head_path is not None:
        h = torch.load(head_path, map_location=device, weights_only=True)
        head = build_fp_head(h["config"])
        head.load_state_dict(h["state_dict"])
        head.eval().to(device)
    return model, head


def load_denovo_v2(path, device=None):
    """Load the v2 generative model for inference (weights fp32 on any device)."""
    device = device or torch.device("cpu")
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = ck["config"]
    model = DeNovoV2Model(cfg["embed_dim"], cfg["enc_layers"], cfg["dec_layers"],
                          cfg["n_heads"], cfg["ff_dim"], vocab_size=cfg["vocab_size"],
                          diff_channel=cfg.get("diff_channel", False))
    model.load_state_dict(ck["state_dict"])
    return model.eval().to(device)


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
    with torch.no_grad():
        probs = torch.sigmoid(head(t)).cpu().numpy()
    return (probs.mean(axis=0) > 0.5).astype(bool)


def generate_molecule_structures(model, tokenizer, spectra, device,
                                 n_samples: int = 8, temperature: float = 0.8,
                                 max_new_tokens: int = 60,
                                 max_structures: int = 64):
    """Sample SMILES structures for a molecule's spectra (third channel).

    spectra: iterable of (mzs, intensities, precursor_mz). Returns unique
    (canonical_smiles, inchikey14, fp_bits) tuples, validity-filtered and
    ordered by generation confidence (best sampled logprob first).
    fp_bits is a 2048-bit boolean Morgan fingerprint (radius 2), matching
    the pool_fps.npz format used by the fingerprint channel.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mz_list, int_list, prec_list = [], [], []
    for spec in spectra:
        mzs, ints = prep_spectrum(spec[0], spec[1])
        if len(mzs) == 0:
            continue
        mz_list.append(mzs)
        int_list.append(ints)
        prec_list.append(float(spec[2]))
    if not mz_list:
        return []
    samples = []
    amp = device.type == "cuda"
    with torch.no_grad():
        for g in range(0, len(mz_list), 8):
            chunk = list(range(g, min(g + 8, len(mz_list))))
            peaks = max(len(mz_list[i]) for i in chunk) + 1
            b = len(chunk)
            mzs = np.zeros((b, peaks), dtype=np.float32)
            ints = np.zeros((b, peaks), dtype=np.float32)
            mask = np.zeros((b, peaks), dtype=np.float32)
            for i, j in enumerate(chunk):
                k = len(mz_list[j]) + 1
                mzs[i, 0] = prec_list[j]
                mzs[i, 1:k] = mz_list[j]
                ints[i, 0] = 2.0
                ints[i, 1:k] = int_list[j]
                mask[i, :k] = 1.0
            mzs_t = torch.from_numpy(mzs).to(device)
            ints_t = torch.from_numpy(ints).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    memory = model.encode(mzs_t, ints_t, mask_t)
                    gen, logprob = model.generate(
                        memory, mask_t, n_samples=n_samples,
                        temperature=temperature, max_new_tokens=max_new_tokens)
            else:
                memory = model.encode(mzs_t, ints_t, mask_t)
                gen, logprob = model.generate(
                    memory, mask_t, n_samples=n_samples,
                    temperature=temperature, max_new_tokens=max_new_tokens)
            for bi in range(gen.shape[0]):
                for si in range(gen.shape[1]):
                    ids = gen[bi, si].tolist()
                    if model.eos_id in ids:
                        ids = ids[: ids.index(model.eos_id)]
                    smiles = tokenizer.decode(
                        [t for t in ids if t not in (0, 1, 2)]).replace(" ", "")
                    if not smiles:
                        continue
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    key = Chem.MolToInchiKey(mol).split("-")[0]
                    if len(key) != 14:
                        continue
                    fp = np.zeros(2048, dtype=bool)
                    for bit in fp_gen.GetFingerprint(mol).GetOnBits():
                        fp[bit] = True
                    samples.append((float(logprob[bi, si]),
                                    Chem.MolToSmiles(mol), key, fp))
    samples.sort(key=lambda t: -t[0])
    out, seen = [], set()
    for _, smiles, key, fp in samples:
        if key in seen:
            continue
        seen.add(key)
        out.append((smiles, key, fp))
        if len(out) >= max_structures:
            break
    return out
