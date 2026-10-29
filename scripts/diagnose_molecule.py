"""Dump full ranking diagnostics for chosen validation molecules.

Runs the honest-validation kernel wiring (same as validate_kernel.py) but
only processes the requested molecules, printing:
  - candidate window size, truth presence in the window/pool
  - spectrum top peaks
  - top-12 ranked candidates with scores and SMILES
  - full feature rows for the truth candidate and the top-1 candidate

Usage:
    uv run python scripts/diagnose_molecule.py cut_0002 cut_0413
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from validate_kernel import build_script  # noqa: E402

MOLS = sys.argv[1:] or ["cut_0002"]
mapping = pd.read_csv(PROJECT_ROOT / "data" / "cut_map.csv")
truth = {r.molecule_id: r.inchikey14 for r in mapping.itertuples(index=False)
         if r.molecule_id in set(MOLS)}

src = build_script(None).read_text(encoding="utf-8")

src = src.replace(
    "results = {}\nfor i, (mid, mol) in enumerate(molecules.items()):",
    f"DIAG_MOLS = {MOLS!r}\nTRUTH = {truth!r}\n"
    "results = {}\nfor i, (mid, mol) in enumerate(molecules.items()):\n"
    "    if mid not in DIAG_MOLS:\n"
    "        continue\n"
    "    print(f'===== DIAG {mid} =====', flush=True)\n"
    "    cand = pool.candidates_for(mol, PPM, PPM_FALLBACK, MAX_CANDIDATES)\n"
    "    if len(cand) == 0:\n"
    "        for wide in (100.0, 500.0):\n"
    "            cand = pool.window(mol.neutral_mass, wide)\n"
    "            if len(cand):\n"
    "                break\n"
    "    print(f'neutral mass {mol.neutral_mass:.4f} | candidates in window: {len(cand)}', flush=True)\n"
    "    tk = TRUTH.get(mid)\n"
    "    ti = pool.key_to_idx.get(tk, -1)\n"
    "    in_cand = int(ti) in set(cand.tolist()) if ti >= 0 else False\n"
    "    print(f'truth {tk}: in pool idx={ti}, in window={in_cand}', flush=True)\n"
    "    for si, spec in enumerate(mol.spectra[:2]):\n"
    "        order = np.argsort(-spec.intensities)[:15]\n"
    "        peaks = '  '.join(f'{spec.mzs[p]:.4f}:{spec.intensities[p]:.2f}' for p in order)\n"
    "        print(f'spectrum {si} ({spec.adduct}, prec {spec.precursor_mz:.4f}): {peaks}', flush=True)\n"
    "    if len(cand) == 0:\n"
    "        continue\n"
    "    sim = engine.sim_scores(mol, cand)\n"
    "    learned = engine.learned_scores(mol, cand)\n"
    "    fused = _zscore(learned) + _zscore(np.maximum(sim - 0.6, 0.0))\n"
    "    ranked = cand[np.argsort(-fused, kind='stable')]\n"
    "    truth_pos = int(np.where(ranked == ti)[0][0]) + 1 if (ti >= 0 and in_cand) else -1\n"
    "    print(f'truth rank in full candidate list: {truth_pos}', flush=True)\n"
    "    print('top-12 ranked:', flush=True)\n"
    "    for rk, ci in enumerate(ranked[:12], 1):\n"
    "        marker = ' <== TRUTH' if ci == ti else ''\n"
    "        print(f'  #{rk:2d} {pool.keys[ci]} fused={fused[np.where(cand==ci)[0][0]]:+.2f} '\n"
    "              f'learned={learned[np.where(cand==ci)[0][0]]:.3f} sim={sim[np.where(cand==ci)[0][0]]:.3f} '\n"
    "              f'dm={pool.masses[ci]-mol.neutral_mass:+.4f} {pool.smiles[ci]}{marker}', flush=True)\n"
    "    from enveda_casmi26.scoring import candidate_row\n"
    "    focus = [ranked[0]]\n"
    "    if ti >= 0 and in_cand and ti != ranked[0]:\n"
    "        focus.append(ti)\n"
    "    if ti >= 0:\n"
    "        print(f'TRUTH SMILES: {pool.smiles[ti]}', flush=True)\n"
    "    eif, learned_mat = engine._frag_matrix(mol, cand)\n"
    "    if ti >= 0 and in_cand:\n"
    "        jt = int(np.where(cand == ti)[0][0])\n"
    "        print(f'truth learned per spectrum: {np.round(learned_mat[:, jt], 3).tolist()}', flush=True)\n"
    "        print(f'truth eif per spectrum: {np.round(eif[:, jt], 3).tolist()}', flush=True)\n"
    "        j1 = int(np.where(cand == ranked[0])[0][0])\n"
    "        print(f'top-1 learned per spectrum: {np.round(learned_mat[:, j1], 3).tolist()}', flush=True)\n"
    "    n_scored = int((~np.isnan(learned_mat)).sum())\n"
    "    print(f'frag-scored cells: {n_scored}/{learned_mat.size}', flush=True)\n"
    "    for ci in focus:\n"
    "        j = int(np.where(cand == ci)[0][0])\n"
    "        label = 'TRUTH' if ci == ti else 'TOP-1 '\n"
    "        print(f'--- features {label} {pool.keys[ci]} ---', flush=True)\n"
    "        for si, spec in enumerate(mol.spectra[:2]):\n"
    "            frag = engine._fragmentor(pool.smiles[ci], spec.adduct)\n"
    "            if frag is None:\n"
    "                continue\n"
    "            row = candidate_row(frag, spec.mzs, spec.intensities, spec.ionization_mode,\n"
    "                                ce_bucket=spec.ce_bucket)\n"
    "            interesting = ['coverage', 'n_matched', 'n_peaks', 'fp_tanimoto', 'w_abs_ppm',\n"
    "                           'w_side_chain_share', 'w_frag_heavy_atoms', 'w_peak_mz_ratio',\n"
    "                           'pair_share', 'ladder_share', 'nested_share', 'w_n_candidates']\n"
    "            vals = '  '.join(f'{k}={row[k]:+.3f}' for k in interesting)\n"
    "            print(f'  spec{si} ({spec.adduct}): {vals}', flush=True)\n",
)

out = Path(r"C:\Users\Michael\AppData\Local\Temp\opencode\kernel_diagnose_script.py")
out.write_text(src, encoding="utf-8")
print(f"script: {out}")

import subprocess  # noqa: E402

proc = subprocess.run([sys.executable, str(out)], cwd=str(PROJECT_ROOT))
sys.exit(proc.returncode)
