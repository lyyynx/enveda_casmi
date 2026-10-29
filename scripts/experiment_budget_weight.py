"""Budget x fused-weight grid experiment on the local validation split.

One expensive pass (MAX_CANDIDATES=500, max_frag_candidates=300) computes the
sim and learned channels once per molecule, then evaluates several fused-score
variants for free:

    fused = zscore(learned) + w * min(zscore(max(sim-0.6, 0)), zcap)

Results are compared to the existing baseline run (cap 200 / frag 100 /
w=1.0) saved in benchmark/validate_results.csv.

Usage: uv run python scripts/experiment_budget_weight.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_kernel import build_script  # noqa: E402

PATCH_HEREDOC = r'''
# ------------------------------------------------- budget x weight experiment
from rdkit import Chem

VARIANTS = [("w1.0", 1.0, None), ("w0.5", 0.5, None), ("w0.25", 0.25, None),
            ("w0", 0.0, None), ("w1.0_zcap2", 1.0, 2.0), ("w0.5_zcap2", 0.5, 2.0)]

mapping = pd.read_csv(find("cut_map.csv"))
TRUTH = dict(zip(mapping["molecule_id"], mapping["inchikey14"]))
SLICE = dict(zip(mapping["molecule_id"], mapping["slice"]))

engine.max_frag_candidates = 300
canon_cache = {}
key_cache = {}

def canon(s):
    if s not in canon_cache:
        m = Chem.MolFromSmiles(s)
        canon_cache[s] = Chem.MolToSmiles(m) if m else s
    return canon_cache[s]

def ikey14(s):
    if s not in key_cache:
        m = Chem.MolFromSmiles(s)
        ik = Chem.MolToInchiKey(m) if m else None
        key_cache[s] = ik[:14] if ik else None
    return key_cache[s]

rows = []
for i, (mid, mol) in enumerate(molecules.items()):
    cand = pool.candidates_for(mol, PPM, PPM_FALLBACK, 500)
    if len(cand) == 0:
        for wide in (100.0, 500.0):
            cand = pool.window(mol.neutral_mass, wide)
            if len(cand):
                break
    if len(cand) == 0:
        for vname, _, _ in VARIANTS:
            rows.append({"molecule_id": mid, "variant": vname, "rr": 0.0, "rank": 0})
        continue
    sim = engine.sim_scores(mol, cand)
    if sim.max() >= GATE_SIM:
        ranked_all = {"gate": cand[np.argsort(-sim, kind="stable")]}
    else:
        learned = engine.learned_scores(mol, cand, sim=sim)
        zl = _zscore(learned)
        zs = _zscore(np.maximum(sim - 0.6, 0.0))
        ranked_all = {}
        for vname, w, zcap in VARIANTS:
            zs_v = zs if zcap is None else np.minimum(zs, zcap)
            fused = zl + w * zs_v
            ranked_all[vname] = cand[np.argsort(-fused, kind="stable")]
    for vname, ranked in ranked_all.items():
        seen, uniq = set(), []
        for ci in ranked:
            s = pool.smiles[ci]
            if s not in seen:
                seen.add(s)
                uniq.append(s)
            if len(uniq) >= N_SLOTS:
                break
        t = TRUTH.get(mid)
        rr, rk = 0.0, 0
        for rank, s in enumerate(uniq[:25], 1):
            if ikey14(canon(s)) == t:
                rr, rk = 1.0 / rank, rank
                break
        rows.append({"molecule_id": mid, "variant": vname, "rr": rr, "rank": rk})
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{n_mol} molecules ({time.time() - T0:.0f}s)", flush=True)

exp = pd.DataFrame(rows)
exp.to_csv(r"EXPROOT/budget_weight_grid.csv", index=False)
print("\n" + "=" * 76)
base = exp[exp.variant == "w1.0"]
for vname, _, _ in VARIANTS:
    g = exp[exp.variant == vname]
    line = (f"{vname:>11}: MRR@25={g['rr'].mean():.4f}  top1={(g['rank'] == 1).mean():.3f}  "
            f"any={(g['rr'] > 0).mean():.3f}  n_flip_vs_w1.0="
            f"{int((base.set_index('molecule_id')['rank'] != g.set_index('molecule_id')['rank']).sum())}")
    print(line, flush=True)
print("=" * 76)
for vname, _, _ in VARIANTS:
    g = exp[exp.variant == vname]
    for sl in ("normal", "defect"):
        gs = g[g.molecule_id.map(SLICE) == sl]
        print(f"{vname:>11} {sl}: n={len(gs):4d}  MRR@25={gs['rr'].mean():.4f}  "
              f"top1={(gs['rank'] == 1).mean():.3f}  any={(gs['rr'] > 0).mean():.3f}", flush=True)
sys.exit(0)
results = {}
for i, (mid, mol) in enumerate(molecules.items()):
    results[mid] = ["CCO"]
'''

src = build_script(None).read_text(encoding="utf-8")
old = "results = {}\nfor i, (mid, mol) in enumerate(molecules.items()):"
assert old in src
patch = PATCH_HEREDOC.replace("EXPROOT", r"C:\Users\Michael\PycharmProjects\kaggle\enveda_casmi26\benchmark")
src = src.replace(old, patch)

out = Path(r"C:\Users\Michael\AppData\Local\Temp\opencode\kernel_budget_weight.py")
out.write_text(src, encoding="utf-8")
print(f"script: {out}", flush=True)
proc = subprocess.run([sys.executable, str(out)], cwd=str(PROJECT_ROOT))
sys.exit(proc.returncode)
