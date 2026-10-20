"""Local smoke test for the submission kernel script.

Patches the kernel script to run against local paths with a small train
subset (first row group) and a molecule cap, then executes it.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Michael\PycharmProjects\kaggle\enveda_casmi26")
KERNEL = ROOT / "kaggle_assets" / "kernel" / "submission_script.py"
WORK = Path(r"C:\Users\Michael\AppData\Local\Temp\opencode\kernel_smoke")

src = KERNEL.read_text(encoding="utf-8")

src = src.replace(
    'WORK = "/kaggle/working"',
    f'WORK = r"{WORK}"',
)
src = src.replace(
    '    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)\n'
    '    return sorted(hits, key=len)[0] if hits else None',
    '    roots = [\n'
    f'        r"{ROOT / "kaggle_assets" / "dataset"}",\n'
    f'        r"{ROOT / "data"}",\n'
    '    ]\n'
    '    hits = [h for r in roots for h in glob.glob(os.path.join(r, name))]\n'
    '    return sorted(hits, key=len)[0] if hits else None',
)
src = src.replace(
    'tab = pq.read_table(TRAIN, columns=["inchikey14", "normalized_smiles", "precursor_mz", "adduct"]).to_pandas()',
    'tab = pq.ParquetFile(TRAIN).read_row_group(0, columns=["inchikey14", "normalized_smiles", "precursor_mz", "adduct"]).to_pandas()',
)
src = src.replace(
    "for batch in pf.iter_batches(batch_size=100_000,\n"
    '                             columns=["inchikey14", "ms2_mzs", "ms2_normalized_intensities",\n'
    '                                      "precursor_mz"]):',
    'for batch in [pq.ParquetFile(TRAIN).read_row_groups([0],\n'
    '        columns=["inchikey14", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz"])]:',
)
src = src.replace(
    "results = {}\nfor i, (mid, mol) in enumerate(molecules.items()):",
    "results = {}\n_mols = list(molecules.items())[:20]\nn_mol = len(_mols)\n"
    "for i, (mid, mol) in enumerate(_mols):",
)

if WORK.exists():
    shutil.rmtree(WORK)
smoke = WORK.parent / "kernel_smoke_script.py"
smoke.write_text(src, encoding="utf-8")
print(f"smoke script: {smoke}")

proc = subprocess.run([sys.executable, str(smoke)], cwd=str(ROOT))
sys.exit(proc.returncode)
