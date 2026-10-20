"""Offline fusion experiments over saved holdout candidate scores.

Usage:
    uv run python scripts/analyze_fusion.py --scores benchmark/holdout_scores_v2.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = PROJECT_ROOT / "benchmark"


def zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if np.all(np.isnan(v)):
        return np.zeros_like(v)
    filled = np.where(np.isnan(v), np.nanmin(v) - 1.0, v)
    std = filled.std()
    if std == 0:
        return np.zeros_like(filled)
    return (filled - filled.mean()) / std


def rr_table(groups: dict, configs: dict) -> None:
    print(f"\n{'config':<28} {'MRR@25':>8} {'top-1':>8} {'top-3':>8} {'top-25':>8} "
          f"{'C2 MRR':>8} {'C2 top1':>9}")
    for name, fn in configs.items():
        rrs, ranks, c2_rrs, c2_ranks = [], [], [], []
        for _, g in groups:
            fused = fn(g)
            order = np.argsort(-fused, kind="stable")
            pos = np.flatnonzero(g["is_true"].to_numpy()[order])[0] + 1
            rrs.append(1.0 / pos if pos <= 25 else 0.0)
            ranks.append(pos)
            if not g["is_class1"].iloc[0]:
                c2_rrs.append(rrs[-1])
                c2_ranks.append(pos)
        c2_rrs = np.array(c2_rrs)
        print(f"{name:<28} {np.mean(rrs):>8.3f} {np.mean(np.array(ranks) == 1):>8.1%} "
              f"{np.mean(np.array(ranks) <= 3):>8.1%} {np.mean(np.array(ranks) <= 25):>8.1%} "
              f"{c2_rrs.mean():>8.3f} {np.mean(np.array(c2_ranks) == 1):>9.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="holdout_scores_v2.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(BENCH_DIR / args.scores)
    print(f"{df.molecule_id.nunique()} molecules, {len(df):,} candidates")
    groups = list(df.groupby("molecule_id"))

    def make_fusion(weights: dict, simclip: float | None = None):
        def fn(g: pd.DataFrame) -> np.ndarray:
            total = np.zeros(len(g))
            for ch, w in weights.items():
                v = g[ch].to_numpy(dtype=float)
                if ch == "sim" and simclip is not None:
                    v = np.maximum(v - simclip, 0.0)
                total += w * zscore(v)
            return total
        return fn

    def make_gate(threshold: float, primary: str, secondary: str | None = None):
        def fn(g: pd.DataFrame) -> np.ndarray:
            sim = g["sim"].to_numpy(dtype=float)
            if sim.max() >= threshold:
                base = zscore(sim)
                return base if secondary is None else base + 0.3 * zscore(g[secondary].to_numpy())
            return zscore(g[primary].to_numpy())
        return fn

    configs = {
        "mass": make_fusion({"mass_delta_ppm": -1}),
        "sim": make_fusion({"sim": 1}),
        "eif": make_fusion({"eif": 1}),
        "learned": make_fusion({"learned": 1}),
        "learned+eif": make_fusion({"learned": 1, "eif": 0.5}),
        "learned+mass": make_fusion({"learned": 1, "mass_delta_ppm": -0.3}),
        "learned+simclip.5": make_fusion({"learned": 1, "sim": 0.5}, simclip=0.5),
        "learned+simclip.7": make_fusion({"learned": 1, "sim": 0.5}, simclip=0.7),
        "learned+simclip.9": make_fusion({"learned": 1, "sim": 1.0}, simclip=0.9),
        "gate.85": make_gate(0.85, "learned"),
        "gate.9": make_gate(0.9, "learned"),
        "gate.95": make_gate(0.95, "learned"),
        "gate.98": make_gate(0.98, "learned"),
        "gate.9+learned": make_gate(0.9, "learned", "learned"),
        "gate.95+learned": make_gate(0.95, "learned", "learned"),
    }
    rr_table(groups, configs)


if __name__ == "__main__":
    main()
