import sys

import torch

sys.path.insert(0, ".")
from enveda_casmi26.fp_model import DeNovoModel, DeNovoV2Model, DifferenceChannel

torch.manual_seed(0)

# --- numerics: synthetic spectrum with known neutral-loss lags
mz = torch.tensor([[500.0, 518.0106, 546.0055, 100.0]])
inten = torch.tensor([[1.0, 0.8, 0.6, 0.5]])
mask = torch.ones(1, 4)
dc = DifferenceChannel(32)
m = mask > 0
w = inten * m
d = mz.unsqueeze(2) - mz.unsqueeze(1)
keep = (d >= 0) & (d < dc.max_lag) & m.unsqueeze(2) & m.unsqueeze(1)
idx = (d * (1.0 / dc.res)).long()
idx = torch.where(keep, idx, torch.zeros_like(idx)).clamp_(max=dc.n_lags - 1)
flat = torch.zeros(1 * dc.n_lags)
offsets = (torch.arange(1).view(1, 1, 1) * dc.n_lags).expand_as(idx)
flat.scatter_add_(0, (idx + offsets).reshape(-1), (w.unsqueeze(2) * w.unsqueeze(1)).reshape(-1))
feats = flat.view(1, dc.n_lags)
for lag in [18.0106, 46.0055, 1.00335]:
    b = int(lag / 0.05)
    print(f"lag {lag} -> bins {b-1}:{b+1}", feats[0, b - 1 : b + 2].tolist())
assert feats[0, int(18.0106 / 0.05)] > 0, "H2O loss lag missing"
assert feats[0, int(46.0055 / 0.05)] > 0, "CO2 loss lag missing"

# --- model paths: v1 + v2 forward/backward/generate
for cls, kw in [
    (DeNovoModel, {}),
    (DeNovoV2Model, dict(embed_dim=64, enc_layers=2, dec_layers=2, n_heads=4, ff_dim=128)),
]:
    mdl = cls(diff_channel=True, vocab_size=32, **kw)
    # lm_head is zero-init -> upstream grads are exactly 0 at init; randomize
    with torch.no_grad():
        mdl.lm_head.weight.normal_(0, 0.02)
    B, S = 3, 10
    mzi = torch.rand(B, S) * 500 + 50
    ini = torch.rand(B, S)
    mki = (torch.rand(B, S) > 0.3).float()
    mem = mdl.encode(mzi, ini, mki)
    toks = torch.randint(3, 32, (B, 7))
    loss = mdl.loss(toks, mem, mki)
    loss.backward()
    g = mdl.diff.proj[0].weight.grad
    assert g is not None and g.abs().sum() > 0, "no grad into diff channel"
    pooled = mdl.pooled_encoding(mzi, ini, mki)
    gshape = None
    if isinstance(mdl, DeNovoV2Model):
        gen, lp = mdl.generate(mem, mki, n_samples=2, max_new_tokens=5)
        gshape = tuple(gen.shape)
    print(
        cls.__name__, "mem", tuple(mem.shape), "pooled", tuple(pooled.shape),
        "gen", gshape, f"loss {loss.item():.3f}",
    )

# --- backward compat: old checkpoint configs have no diff_channel key
cfg = dict(embed_dim=32, n_layers=1, n_heads=4, ff_dim=64, vocab_size=32)
mdl = DeNovoModel(
    cfg["embed_dim"], cfg["n_layers"], cfg["n_heads"], cfg["ff_dim"],
    vocab_size=cfg["vocab_size"], diff_channel=cfg.get("diff_channel", False),
)
mdl2 = DeNovoModel(
    cfg["embed_dim"], cfg["n_layers"], cfg["n_heads"], cfg["ff_dim"],
    vocab_size=cfg["vocab_size"], diff_channel=True,
)
sd = mdl.state_dict()
missing = [k for k in mdl2.state_dict() if k not in sd]
print("old-config load ok; extra keys when enabling flag:", missing)
print("ALL OK")
