"""Run the submission kernel against the local validation split and score it.

Executes the real kernel code path with TRAIN=train_cut.parquet,
TEST=test_cut.parquet and the external pool = coconut + held-out structures
(val_external_pool.parquet), then scores MRR@25 on inchikey14 by slice.

This is the honest pre-submit estimate: truth structures are absent from
train_cut spectra AND pool, exactly like Class-2 molecules whose structure
lives in an external database.

Usage:
    uv run python scripts/validate_kernel.py [--n-mols 150] [--keep-script]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
KERNEL = PROJECT_ROOT / "kaggle_assets" / "kernel" / "submission_script.py"
DATA = PROJECT_ROOT / "data"
WORK = Path(r"C:\Users\Michael\AppData\Local\Temp\opencode\kernel_validate")


def build_script(n_mols: int | None, scorer: str = "scorer_default.json") -> Path:
    src = KERNEL.read_text(encoding="utf-8")
    src = src.replace('WORK = "/kaggle/working"', f'WORK = r"{WORK}"')
    src = src.replace('find("scorer_default.json")', f'find("{scorer}")')
    src = src.replace(
        '    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)\n'
        '    return sorted(hits, key=len)[0] if hits else None',
        '    roots = [\n'
        f'        r"{PROJECT_ROOT / "kaggle_assets" / "dataset"}",\n'
        f'        r"{DATA}",\n'
        '    ]\n'
        '    hits = [h for r in roots for h in glob.glob(os.path.join(r, name))]\n'
        '    return sorted(hits, key=len)[0] if hits else None',
    )
    src = src.replace(
        'COMP = os.path.dirname(find("test.parquet"))\n'
        'TRAIN = os.path.join(COMP, "train.parquet")\n'
        'TEST = os.path.join(COMP, "test.parquet")\n'
        'SAMPLE = os.path.join(COMP, "sample_submission.csv")',
        f'COMP = r"{DATA}"\n'
        'TRAIN = os.path.join(COMP, "train_cut.parquet")\n'
        'TEST = os.path.join(COMP, "test_cut.parquet")\n'
        'SAMPLE = ""',
    )
    src = src.replace('find("coconut_pool.parquet")', 'find("val_external_pool.parquet")')
    if n_mols is not None:
        src = src.replace(
            "results = {}\nfor i, (mid, mol) in enumerate(molecules.items()):",
            f"results = {{}}\n_mols = list(molecules.items())[:{n_mols}]\nn_mol = len(_mols)\n"
            "for i, (mid, mol) in enumerate(_mols):",
        )
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.parent.mkdir(parents=True, exist_ok=True)
    script = WORK.parent / "kernel_validate_script.py"
    script.write_text(src, encoding="utf-8")
    return script


def score(sub_path: Path) -> None:
    from rdkit import Chem

    sub = pd.read_csv(sub_path)
    mapping = pd.read_csv(DATA / "cut_map.csv")
    truth = dict(zip(mapping["molecule_id"], mapping["inchikey14"]))
    slice_of = dict(zip(mapping["molecule_id"], mapping["slice"]))

    cache: dict[str, str | None] = {}

    def ikey14(smiles: str) -> str | None:
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles)
            ik = Chem.MolToInchiKey(mol) if mol else None
            cache[smiles] = ik[:14] if ik else None
        return cache[smiles]

    rows = []
    for _, r in sub.iterrows():
        t = truth.get(r["molecule_id"])
        if t is None:
            continue
        keys = [ikey14(s) for s in str(r["smiles"]).split(";")]
        rr, hit_rank = 0.0, 0
        for rank, k in enumerate(keys[:25], start=1):
            if k == t:
                rr, hit_rank = 1.0 / rank, rank
                break
        rows.append({"molecule_id": r["molecule_id"], "slice": slice_of[r["molecule_id"]],
                     "rr": rr, "top1": hit_rank == 1, "hit": rr > 0,
                     "rank": hit_rank})

    res = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    for sl, g in list(res.groupby("slice")) + [("", res)]:
        name = sl if sl else "ALL"
        print(f"{name:>7}: n={len(g):4d}  MRR@25={g['rr'].mean():.4f}  "
              f"top-1={g['top1'].mean():.3f}  any={g['hit'].mean():.3f}")
    res.to_csv(PROJECT_ROOT / "benchmark" / "validate_results.csv", index=False)
    print("per-molecule details: benchmark/validate_results.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-mols", type=int, default=None,
                        help="cap molecules for a quicker iteration run")
    parser.add_argument("--scorer", default="scorer_default.json",
                        help="scorer JSON file name to load")
    parser.add_argument("--keep-script", action="store_true")
    args = parser.parse_args()

    script = build_script(args.n_mols, args.scorer)
    print(f"validation script: {script}")
    print(f"work dir: {WORK}\n")
    proc = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    score(WORK / "submission.csv")


if __name__ == "__main__":
    main()
