"""Tier-4 likelihood scoring at the candidate level.

Per-match features barely separate true-structure matches from decoy
coincidences; the discriminating signal is *how much* of a spectrum a
candidate explains and the chemical *quality* of those explanations. Each
candidate structure therefore becomes one row of aggregated features, and a
logistic model learns P(candidate is the true structure | features).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NUMERIC_FEATURES = [
    "coverage",
    "n_matched",
    "n_peaks",
    "w_abs_ppm",
    "w_side_chain_share",
    "w_ring_fission_share",
    "w_precursor_share",
    "w_noloss_share",
    "w_frag_heavy_atoms",
    "w_cut_bond_aromatic",
    "w_cut_bond_max_order",
    "w_peak_mz_ratio",
    "w_h_shift",
    "w_n_candidates",
    "pair_share",
    "ladder_share",
    "nested_share",
    "fp_tanimoto",
    "ce_prec_dev",
    "ce_side_dev",
    "ce_ring_dev",
    "ce_noloss_dev",
    "ce_h0_dev",
]
CATEGORICAL_FEATURES = ["adduct", "ionization_mode"]

# Collision-energy-dependent fragmentation priors, mined from train
# (scripts/mine_ce_priors.py): per CE bucket, the share of total spectral
# intensity explained by each cleavage kind / no-loss matches / h-shift-0
# fragments when the *true* structure is matched.
CE_PRIORS = {
    # bucket: 0 unknown, 1 low, 2 medium, 3 high
    0: {"prec": 0.074, "side": 0.202, "ring": 0.148, "noloss": 0.239, "h0": 0.125},
    1: {"prec": 0.127, "side": 0.228, "ring": 0.164, "noloss": 0.339, "h0": 0.197},
    2: {"prec": 0.362, "side": 0.191, "ring": 0.137, "noloss": 0.488, "h0": 0.381},
    3: {"prec": 0.124, "side": 0.229, "ring": 0.202, "noloss": 0.299, "h0": 0.167},
}


def _consistency_features(matches: pd.DataFrame, weights: np.ndarray) -> tuple[float, float, float]:
    """Intensity shares of matches participating in fragment correlations.

    pair_share   -- both sides of one cut observed (complementary ion pair)
    ladder_share -- a fragment and its neutral-loss sibling both observed
    nested_share -- a fragment nested inside another observed fragment (same
                    substructure family, e.g. ring-only vs ring+side)
    """
    sides = [tuple(s) for s in matches["side_atoms"]]
    masks = []
    for s in sides:
        m = 0
        for a in s:
            m |= 1 << a
        masks.append(m)
    cuts = matches["cut_bonds"].astype(str).to_numpy()
    losses = matches["neutral_loss"].astype(str).to_numpy()
    hs = matches["h_shift"].to_numpy()
    n = len(matches)
    linked_pair = np.zeros(n, dtype=bool)
    linked_nested = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if masks[i] & masks[j]:
                if (masks[i] | masks[j]) in (masks[i], masks[j]) and masks[i] != masks[j]:
                    linked_nested[i] = linked_nested[j] = True
            elif cuts[i] == cuts[j]:
                linked_pair[i] = linked_pair[j] = True
    no_loss_keys = {(sides[i], int(hs[i])) for i in range(n) if losses[i] == "none"}
    loss_keys = {(sides[i], int(hs[i])) for i in range(n) if losses[i] != "none"}
    ladder_keys = no_loss_keys & loss_keys
    linked_ladder = np.array([(sides[i], int(hs[i])) in ladder_keys for i in range(n)],
                             dtype=bool)
    wsum = float(weights.sum())
    if wsum <= 0:
        return 0.0, 0.0, 0.0
    return (float(weights[linked_pair].sum()),
            float(weights[linked_ladder].sum()),
            float(weights[linked_nested].sum()))


def candidate_features(
    matches: pd.DataFrame,
    n_peaks: int,
    precursor_mz: float,
    adduct: str,
    ionization_mode: str,
    total_intensity: float,
    ce_bucket: int = 0,
    fp_tanimoto: float = 0.0,
) -> dict:
    """Aggregate a candidate's best-per-peak matches into one feature row."""
    prior = CE_PRIORS.get(ce_bucket, CE_PRIORS[0])
    if matches.empty or total_intensity <= 0:
        return {
            "coverage": 0.0, "n_matched": 0, "n_peaks": n_peaks,
            "w_abs_ppm": 0.0, "w_side_chain_share": 0.0, "w_ring_fission_share": 0.0,
            "w_precursor_share": 0.0, "w_noloss_share": 0.0, "w_frag_heavy_atoms": 0.0,
            "w_cut_bond_aromatic": 0.0, "w_cut_bond_max_order": 0.0, "w_peak_mz_ratio": 0.0,
            "w_h_shift": 0.0, "w_n_candidates": 0.0, "fp_tanimoto": fp_tanimoto,
            "pair_share": 0.0, "ladder_share": 0.0, "nested_share": 0.0,
            "ce_prec_dev": 0.0, "ce_side_dev": 0.0, "ce_ring_dev": 0.0,
            "ce_noloss_dev": 0.0, "ce_h0_dev": 0.0,
            "adduct": adduct, "ionization_mode": ionization_mode,
        }

    w = matches["intensity"].to_numpy(dtype=float)
    wsum = w.sum()
    weights = w / wsum if wsum > 0 else np.full(len(w), 1.0 / len(w))

    def wmean(col):
        return float(np.dot(weights, matches[col].to_numpy(dtype=float)))

    kind = matches["kind"].to_numpy()
    loss = matches["neutral_loss"].to_numpy()
    h_shift = matches["h_shift"].to_numpy()
    side_share = float(weights[kind == "side_chain"].sum())
    ring_share = float(weights[kind == "ring_fission"].sum())
    prec_share = float(weights[kind == "precursor"].sum())
    noloss_share = float(weights[kind == "none"].sum())
    h0_share = float(weights[h_shift == 0].sum())
    pair_share, ladder_share, nested_share = _consistency_features(matches, weights)
    return {
        "coverage": float(wsum / total_intensity),
        "n_matched": int(len(matches)),
        "n_peaks": int(n_peaks),
        "w_abs_ppm": float(np.dot(weights, matches["error_ppm"].to_numpy(dtype=float).__abs__())),
        "w_side_chain_share": side_share,
        "w_ring_fission_share": ring_share,
        "w_precursor_share": prec_share,
        "w_noloss_share": noloss_share,
        "w_frag_heavy_atoms": wmean("frag_heavy_atoms"),
        "w_cut_bond_aromatic": wmean("cut_bond_aromatic"),
        "w_cut_bond_max_order": wmean("cut_bond_max_order"),
        "w_peak_mz_ratio": wmean("peak_mz") / precursor_mz,
        "w_h_shift": wmean("h_shift"),
        "w_n_candidates": wmean("n_candidates_at_peak"),
        "pair_share": pair_share,
        "ladder_share": ladder_share,
        "nested_share": nested_share,
        "fp_tanimoto": fp_tanimoto,
        "ce_prec_dev": prec_share - prior["prec"],
        "ce_side_dev": side_share - prior["side"],
        "ce_ring_dev": ring_share - prior["ring"],
        "ce_noloss_dev": noloss_share - prior["noloss"],
        "ce_h0_dev": h0_share - prior["h0"],
        "adduct": adduct,
        "ionization_mode": ionization_mode,
    }


def candidate_row(fragmentor, mzs, intensities, ionization_mode: str,
                 ce_bucket: int = 0, fp_tanimoto: float = 0.0) -> dict:
    """Build one candidate's feature row from its fragmentor and a spectrum."""
    intensities = np.asarray(intensities, dtype=float)
    matches = fragmentor.match_peaks(mzs, intensities)
    return candidate_features(
        matches,
        n_peaks=len(mzs),
        precursor_mz=fragmentor.precursor_mz,
        adduct=fragmentor.adduct,
        ionization_mode=ionization_mode,
        total_intensity=float(intensities.sum()),
        ce_bucket=ce_bucket,
        fp_tanimoto=fp_tanimoto,
    )


class CandidateScorer:
    """Logistic model over within-spectrum contrasted candidate features.

    The discriminating signal is how a candidate compares to the *other
    candidates for the same spectrum*, not its absolute feature values, so
    features are z-scored within each candidate set before learning.
    """

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.columns: list[str] | None = None
        self.num_mean: pd.Series | None = None
        self.num_std: pd.Series | None = None

    @staticmethod
    def contrast_within_set(rows: pd.DataFrame) -> pd.DataFrame:
        x = rows[NUMERIC_FEATURES].astype(float)
        mean = x.mean()
        std = x.std(ddof=0).replace(0.0, 1.0)
        return (x - mean) / std

    def _featurize(self, contrasted: pd.DataFrame) -> pd.DataFrame:
        x = contrasted.copy()
        for col in NUMERIC_FEATURES:
            x[col] = (x[col] - self.num_mean[col]) / self.num_std[col]
        return x.reindex(columns=self.columns, fill_value=0.0)

    def fit(self, contrasted: pd.DataFrame, labels: np.ndarray) -> None:
        self.num_mean = contrasted[NUMERIC_FEATURES].mean()
        self.num_std = contrasted[NUMERIC_FEATURES].std(ddof=0).replace(0.0, 1.0)
        x = self._featurize(contrasted)
        self.columns = list(x.columns)
        self.model = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
        self.model.fit(x, labels)

    def score_set(self, rows: list[dict]) -> np.ndarray:
        """P(true) for each candidate of one spectrum, given all candidates' rows."""
        contrasted = self.contrast_within_set(pd.DataFrame(rows))
        return 1.0 - self.model.predict_proba(self._featurize(contrasted))[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "columns": self.columns,
                "num_mean": self.num_mean,
                "num_std": self.num_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "CandidateScorer":
        bundle = joblib.load(path)
        scorer = cls()
        scorer.model = bundle["model"]
        scorer.columns = bundle["columns"]
        scorer.num_mean = bundle["num_mean"]
        scorer.num_std = bundle["num_std"]
        return scorer
