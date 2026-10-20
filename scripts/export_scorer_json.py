"""Export the trained CandidateScorer to plain JSON (no pickle dependencies)."""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from enveda_casmi26.scoring import NUMERIC_FEATURES, CandidateScorer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=str(PROJECT_ROOT / "models/scorer.joblib"))
parser.add_argument("--out", default=str(PROJECT_ROOT / "models/scorer_default.json"))
args = parser.parse_args()

scorer = CandidateScorer.load(args.model)

bundle = {
    "coef": scorer.model.coef_[0].tolist(),
    "intercept": float(scorer.model.intercept_[0]),
    "columns": list(scorer.columns),
    "num_mean": {c: float(scorer.num_mean[c]) for c in scorer.num_mean.index},
    "num_std": {c: float(scorer.num_std[c]) for c in scorer.num_std.index},
    "numeric_features": list(scorer.num_mean.index),
}
out = Path(args.out)
out.write_text(json.dumps(bundle, indent=1))
print(f"wrote {out}")
print(f"features: {len(bundle['columns'])}, coef range: "
      f"[{min(bundle['coef']):.2f}, {max(bundle['coef']):.2f}]")
