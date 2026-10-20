"""Mine collision-energy-dependent fragmentation priors from train.

For a sample of train spectra per CE bucket, match the spectrum against its
own structure with the default Fragmentor and aggregate:

  * intensity share of explained peaks by cleavage kind (side_chain /
    ring_fission / precursor) per CE bucket
  * intensity-weighted neutral-loss frequency per CE bucket
  * h-shift distribution per CE bucket
  * overall explained-intensity fraction per CE bucket

CE bucketing (mix of eV / NCE / V / unknown units):
  0 unknown   1 low   2 medium   3 high
with unit-specific thresholds chosen from the train distribution.

Output: benchmark/ce_priors.csv (long format) + console summary.

Usage:
    uv run python scripts/mine_ce_priors.py [--per-bucket 150]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from enveda_casmi26.cleaning import CleanConfig, SpectrumCleaner  # noqa: E402
from enveda_casmi26.fragmentation import Fragmentor  # noqa: E402

DATA = PROJECT_ROOT / "data"


def ce_bucket(value, unit: str) -> int:
    """0 unknown, 1 low, 2 medium, 3 high."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(value):
        return 0
    unit = (unit or "unknown").strip()
    if unit == "eV":
        return 1 if value < 15 else (2 if value < 35 else 3)
    if unit == "NCE":
        return 1 if value < 20 else (2 if value < 45 else 3)
    if unit == "V":
        return 1 if value < 25 else (2 if value < 50 else 3)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-bucket", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print("collecting spectra per CE bucket ...")
    pf = pq.ParquetFile(DATA / "train.parquet")
    per_bucket = defaultdict(list)
    for batch in pf.iter_batches(batch_size=100_000,
                                 columns=["inchikey14", "normalized_smiles", "adduct",
                                          "precursor_mz", "ms2_mzs",
                                          "ms2_normalized_intensities",
                                          "collision_energy_orig",
                                          "collision_energy_orig_units"]):
        for k, smi, ad, pm, mzs, ints, ce, un in zip(
                batch.column("inchikey14").to_pylist(),
                batch.column("normalized_smiles").to_pylist(),
                batch.column("adduct").to_pylist(),
                batch.column("precursor_mz").to_pylist(),
                batch.column("ms2_mzs").to_pylist(),
                batch.column("ms2_normalized_intensities").to_pylist(),
                batch.column("collision_energy_orig").to_pylist(),
                batch.column("collision_energy_orig_units").to_pylist()):
            if k is None or smi is None or not mzs:
                continue
            b = ce_bucket(ce, un)
            if len(per_bucket[b]) < args.per_bucket * 3:
                per_bucket[b].append((k, smi, ad, pm, mzs, ints, ce, un))
    for b, rows in sorted(per_bucket.items()):
        pick = rng.choice(len(rows), size=min(args.per_bucket, len(rows)),
                         replace=False)
        per_bucket[b] = [rows[i] for i in pick]
        print(f"  bucket {b}: sampled {len(per_bucket[b])} spectra")

    cleaner = SpectrumCleaner(CleanConfig())
    records = []
    for b, rows in sorted(per_bucket.items()):
        kind_int = defaultdict(float)
        loss_int = defaultdict(float)
        h_int = defaultdict(float)
        expl = []
        for k, smi, ad, pm, mzs, ints, ce, un in rows:
            try:
                frag = Fragmentor(smi, adduct=ad, tolerance_ppm=10.0)
            except Exception:
                continue
            res = cleaner.clean(np.asarray(mzs, dtype=float),
                                np.asarray(ints, dtype=float),
                                precursor_mz=pm)
            if res.n_kept == 0:
                continue
            m = frag.match_peaks(res.mzs, res.intensities)
            if m.empty:
                continue
            tot = float(res.intensities.sum())
            for _, r in m.iterrows():
                kind_int[r["kind"]] += r["intensity"] / tot
                loss_int[r["neutral_loss"]] += r["intensity"] / tot
                h_int[r["h_shift"]] += r["intensity"] / tot
            expl.append(m["intensity"].sum() / tot)
        n = max(len(expl), 1)
        for kind, v in kind_int.items():
            records.append({"bucket": b, "stat": f"kind_share:{kind}",
                            "value": v / n, "n": len(expl)})
        for loss, v in sorted(loss_int.items(), key=lambda x: -x[1])[:8]:
            records.append({"bucket": b, "stat": f"loss_share:{loss}",
                            "value": v / n, "n": len(expl)})
        for h, v in sorted(h_int.items()):
            records.append({"bucket": b, "stat": f"h_shift:{h}",
                            "value": v / n, "n": len(expl)})
        records.append({"bucket": b, "stat": "explained_intensity",
                        "value": float(np.mean(expl)) if expl else 0.0,
                        "n": len(expl)})
        print(f"bucket {b}: n={len(expl)}  explained={np.mean(expl) if expl else 0:.1%}")

    out = pd.DataFrame(records)
    out.to_csv(PROJECT_ROOT / "benchmark" / "ce_priors.csv", index=False)

    print("\nintensity share by cleavage kind:")
    piv = out[out["stat"].str.startswith("kind_share")].pivot(
        index="stat", columns="bucket", values="value")
    print(piv.to_string())
    print("\ntop losses by bucket (share of total intensity):")
    ls = out[out["stat"].str.startswith("loss_share")]
    for b in sorted(ls["bucket"].unique()):
        top = ls[ls["bucket"] == b].nlargest(4, "value")
        print(f"  bucket {b}: " + ", ".join(
            f"{r['stat'].split(':')[1]} {r['value']:.1%}" for _, r in top.iterrows()))
    print("\nh-shift distribution:")
    hs = out[out["stat"].str.startswith("h_shift")].pivot(
        index="stat", columns="bucket", values="value")
    print(hs.fillna(0).to_string())


if __name__ == "__main__":
    main()
