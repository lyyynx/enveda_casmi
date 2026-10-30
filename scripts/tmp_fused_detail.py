"""Detailed fused-score decomposition for cut_0086 with the raised budget."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_kernel import build_script  # noqa: E402

src = build_script(None).read_text(encoding="utf-8")
src = src.replace(
    "results = {}\nfor i, (mid, mol) in enumerate(molecules.items()):",
    'mol = molecules["cut_0086"]\n'
    'cand = pool.candidates_for(mol, PPM, PPM_FALLBACK, 500)\n'
    'engine.max_frag_candidates = 300\n'
    'mapping = pd.read_csv(find("cut_map.csv"))\n'
    'tk = dict(zip(mapping.molecule_id, mapping.inchikey14))["cut_0086"]\n'
    'ti = pool.key_to_idx.get(tk, -1)\n'
    'sim = engine.sim_scores(mol, cand)\n'
    'learned = engine.learned_scores(mol, cand, sim=sim)\n'
    'simclip = np.maximum(sim - 0.6, 0.0)\n'
    'zl = _zscore(learned)\n'
    'zs = _zscore(simclip)\n'
    'fused = zl + zs\n'
    'ranked = cand[np.argsort(-fused, kind="stable")]\n'
    'print(f"n_candidates={len(cand)}, n with sim>0.6: {(sim>0.6).sum()}, "\n'
    '      f"n with sim>0: {(sim>0).sum()}, n learned-scored: {np.isfinite(learned).sum()}", flush=True)\n'
    'for rk, ci in enumerate(ranked[:8], 1):\n'
    '    j = int(np.where(cand == ci)[0][0])\n'
    '    mark = " <== TRUTH" if ci == ti else ""\n'
    '    print(f"#{rk} learned={learned[j]:.4f} z(learned)={zl[j]:+.2f} sim={sim[j]:.3f} "\n'
    '          f"z(simclip)={zs[j]:+.2f} fused={fused[j]:+.2f} {pool.smiles[ci][:48]}{mark}", flush=True)\n'
    'jt = int(np.where(cand == ti)[0][0])\n'
    'print(f"TRUTH: learned={learned[jt]:.4f} z(learned)={zl[jt]:+.2f} sim={sim[jt]:.3f} "\n'
    '      f"z(simclip)={zs[jt]:+.2f} fused={fused[jt]:+.2f}", flush=True)\n'
    'sys.exit(0)\n'
    'results = {}\n'
    'for i, (mid, mol) in enumerate(molecules.items()):\n'
    '    results[mid] = ["CCO"]\n',
)

out = Path(r"C:\Users\Michael\AppData\Local\Temp\opencode\kernel_fused_detail.py")
out.write_text(src, encoding="utf-8")
print(f"script: {out}", flush=True)
proc = subprocess.run([sys.executable, str(out)], cwd=str(PROJECT_ROOT))
sys.exit(proc.returncode)
