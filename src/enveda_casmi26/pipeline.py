"""Competition pipeline: candidate retrieval + multi-channel ranking.

Simulates (and later runs) the real task: for each test molecule --
a group of MS2 spectra sharing precursor m/z and adducts -- retrieve
candidate structures by neutral-mass window and rank them with
evidence channels:

  mass   proximity of the candidate's neutral mass to the query
  sim    spectral similarity against the candidate's reference spectra
  eif    in-silico fragmentation explained-intensity fraction
  learned  CandidateScorer P(true) from contrasted candidate features

Scores are z-scored within each molecule's candidate set and summed
per configuration. Ranking metric: MRR@25 on inchikey14.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", message="Mean of empty slice")

import numpy as np
import pandas as pd

from enveda_casmi26.fragmentation import ADDUCTS, ADDUCT_ALIASES


def neutral_mass(precursor_mz: float, adduct: str) -> float:
    """Neutral monoisotopic mass implied by a precursor m/z and adduct."""
    base = ADDUCT_ALIASES.get(adduct, adduct)
    if base not in ADDUCTS or not np.isfinite(precursor_mz):
        return math.nan
    delta, charge = ADDUCTS[base]
    mass = precursor_mz * charge - delta
    if adduct.startswith("[2M"):
        mass /= 2.0
    return mass


def ce_bucket(value, unit: str | None) -> int:
    """Coarse collision-energy bucket: 0 unknown, 1 low, 2 medium, 3 high.

    Thresholds are unit-specific (eV / NCE / V) and chosen from the train
    distribution (see scripts/mine_ce_priors.py).
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(value):
        return 0
    unit = (unit or "").strip()
    if unit == "eV":
        return 1 if value < 15 else (2 if value < 35 else 3)
    if unit == "NCE":
        return 1 if value < 20 else (2 if value < 45 else 3)
    if unit == "V":
        return 1 if value < 25 else (2 if value < 50 else 3)
    return 0


def refined_cosine(mzs_a, ints_a, mzs_b, ints_b, tol: float = 0.02) -> float:
    na = float(np.linalg.norm(ints_a))
    nb = float(np.linalg.norm(ints_b))
    if na == 0 or nb == 0:
        return 0.0
    i = j = 0
    dot = 0.0
    while i < len(mzs_a) and j < len(mzs_b):
        d = mzs_a[i] - mzs_b[j]
        if abs(d) <= tol:
            dot += ints_a[i] * ints_b[j]
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return dot / (na * nb)


@dataclass
class MoleculeSpectrum:
    mzs: np.ndarray
    intensities: np.ndarray
    adduct: str
    precursor_mz: float
    ionization_mode: str
    ce_bucket: int = 0


@dataclass
class Molecule:
    molecule_id: str
    spectra: list[MoleculeSpectrum] = field(default_factory=list)
    mass_noise_ppm: float = 0.0
    true_key: str = ""

    @property
    def neutral_mass(self) -> float:
        masses = [neutral_mass(s.precursor_mz, s.adduct) for s in self.spectra]
        masses = [m for m in masses if np.isfinite(m)]
        if not masses:
            return math.nan
        mass = float(np.median(masses))
        if self.mass_noise_ppm:
            mass *= 1.0 + self.mass_noise_ppm * 1e-6
        return mass


class CandidatePool:
    """Unique structures sorted by neutral mass for windowed retrieval."""

    def __init__(self, structures: pd.DataFrame) -> None:
        """structures: inchikey14, normalized_smiles, neutral_mass (unique keys)."""
        structures = structures.dropna(subset=["neutral_mass"]).sort_values("neutral_mass")
        self.keys = structures["inchikey14"].to_numpy()
        self.smiles = structures["normalized_smiles"].to_numpy()
        self.masses = structures["neutral_mass"].to_numpy(dtype=float)
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys)

    def window(self, mass: float, ppm: float) -> np.ndarray:
        if not np.isfinite(mass):
            return np.array([], dtype=np.int64)
        tol = ppm * 1e-6 * mass
        lo = np.searchsorted(self.masses, mass - tol, side="left")
        hi = np.searchsorted(self.masses, mass + tol, side="right")
        return np.arange(lo, hi)

    def candidates_for(self, molecule: Molecule, ppm: float, ppm_fallback: float,
                       max_candidates: int) -> np.ndarray:
        idx = self.window(molecule.neutral_mass, ppm)
        if len(idx) == 0:
            idx = self.window(molecule.neutral_mass, ppm_fallback)
        if len(idx) > max_candidates:
            deltas = np.abs(self.masses[idx] - molecule.neutral_mass)
            idx = idx[np.argsort(deltas, kind="stable")[:max_candidates]]
        return idx


class SpectralLibrary:
    """Reference spectra per structure, excluding given keys or exact rows.

    Accepts multiple spectra per structure (e.g. one best spectrum per CE
    bucket). At query time the spectrum whose CE bucket best matches the
    query's is used for the cosine.
    """

    def __init__(self, df: pd.DataFrame, exclude_keys: set[str] | None = None,
                 exclude_rows: set[int] | None = None) -> None:
        """df needs inchikey14, ms2_mzs, ms2_normalized_intensities,
        num_peaks_clean, drop_reason; optional ce_bucket. Multiple rows per
        key are allowed; the best (clean, most peaks) row per CE bucket is
        kept."""
        df = df[~df["inchikey14"].isin(exclude_keys or ())]
        df = df[df["inchikey14"].notna()].copy()
        if exclude_rows:
            keep = np.ones(len(df), dtype=bool)
            for r in exclude_rows:
                keep[r] = False
            df = df[keep]
        if "ce_bucket" not in df.columns:
            df["ce_bucket"] = 0
        df["_ok"] = (df["drop_reason"] == "").astype(int)
        df = df.sort_values(["inchikey14", "_ok", "num_peaks_clean"],
                            ascending=[True, False, False])
        self.keys = np.asarray(sorted(df["inchikey14"].unique()), dtype=object)
        self._specs: dict[str, list[tuple[int, np.ndarray, np.ndarray, int]]] = {}
        for r in df.itertuples(index=False):
            entry = (int(r.ce_bucket), np.asarray(r.ms2_mzs, dtype=float),
                     np.asarray(r.ms2_normalized_intensities, dtype=float),
                     int(r.num_peaks_clean))
            specs = self._specs.setdefault(r.inchikey14, [])
            if entry[0] not in [s[0] for s in specs]:
                specs.append(entry)

    def __len__(self) -> int:
        return len(self.keys)

    def cosine(self, key: str, mzs: np.ndarray, ints: np.ndarray,
               exclude_arrays: list | None = None,
               query_bucket: int = 0) -> float:
        """Cosine vs the structure's best CE-matched spectrum; skips spectra
        that are exact copies of the query.

        Bucket matching is quality-gated: a bucket-matched spectrum is only
        used if it carries at least half the peak count of the structure's
        best spectrum, otherwise the best spectrum overall is preferred
        (bucket-matched library entries are often sparse)."""
        specs = self._specs.get(key)
        if not specs:
            return 0.0
        qb = int(query_bucket or 0)
        if qb:
            key_max = max(s[3] for s in specs)
            good = [i for i in range(len(specs))
                    if specs[i][0] and specs[i][3] >= 0.5 * key_max]
            order = sorted(good,
                           key=lambda i: (abs(specs[i][0] - qb), -specs[i][3]))
            order += [i for i in sorted(range(len(specs)), key=lambda i: -specs[i][3])
                      if i not in set(order)]
        else:
            order = sorted(range(len(specs)), key=lambda i: -specs[i][3])
        for i in order:
            rm, ri = specs[i][1], specs[i][2]
            if exclude_arrays is not None:
                skip = False
                for em, ei in exclude_arrays:
                    if np.array_equal(rm, em) and np.array_equal(ri, ei):
                        skip = True
                        break
                if skip:
                    continue
            return refined_cosine(mzs, ints, rm, ri)
        return 0.0


def _zscore(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if np.all(np.isnan(scores)):
        return np.zeros_like(scores)
    # candidates with no evidence rank below every scored candidate
    filled = np.where(np.isnan(scores), np.nanmin(scores) - 1.0, scores)
    std = filled.std()
    if std == 0:
        return np.zeros_like(filled)
    return (filled - filled.mean()) / std


class ChannelEngine:
    """Computes per-candidate evidence scores for one molecule."""

    def __init__(self, pool: CandidatePool, library: SpectralLibrary,
                 scorer=None, max_frag_candidates: int = 60,
                 fp_predictor=None, candidate_fp_lookup=None) -> None:
        self.pool = pool
        self.library = library
        self.scorer = scorer
        self.max_frag_candidates = max_frag_candidates
        self.fp_predictor = fp_predictor
        self.candidate_fp_lookup = candidate_fp_lookup
        self._frag_cache: dict[tuple[str, str], object] = {}

    def mass_scores(self, cand_idx: np.ndarray, query_mass: float) -> np.ndarray:
        return -np.abs(self.pool.masses[cand_idx] - query_mass)

    def sim_scores(self, molecule: Molecule, cand_idx: np.ndarray) -> np.ndarray:
        keys = self.pool.keys[cand_idx]
        out = np.zeros(len(cand_idx))
        exclude = None
        if molecule.true_key:
            exclude = [(s.mzs, s.intensities) for s in molecule.spectra]
        for spec in molecule.spectra:
            for j, key in enumerate(keys):
                out[j] = max(out[j], self.library.cosine(
                    key, spec.mzs, spec.intensities,
                    exclude_arrays=exclude if key == molecule.true_key else None,
                    query_bucket=spec.ce_bucket))
        return out

    def _fragmentor(self, smiles: str, adduct: str):
        from enveda_casmi26.fragmentation import Fragmentor

        ck = (smiles, adduct)
        if ck not in self._frag_cache:
            try:
                self._frag_cache[ck] = Fragmentor(smiles, adduct=adduct, tolerance_ppm=10.0)
            except Exception:
                self._frag_cache[ck] = None
            if len(self._frag_cache) > 40000:
                self._frag_cache.clear()
        return self._frag_cache[ck]

    def _frag_matrix(self, molecule: Molecule, cand_idx: np.ndarray):
        """ EIF matrix and learned P(true) matrix, both (n_spectra, n_candidates)."""
        from enveda_casmi26.scoring import candidate_row

        keys = self.pool.keys[cand_idx]
        smiles = self.pool.smiles[cand_idx]
        n_c, n_s = len(cand_idx), len(molecule.spectra)
        eif = np.full((n_s, n_c), np.nan)
        learned = np.full((n_s, n_c), np.nan)
        order = np.argsort(np.abs(self.pool.masses[cand_idx] - molecule.neutral_mass))
        frag_positions = set(order[: self.max_frag_candidates].tolist())
        cand_tani = np.zeros(n_c)
        if self.fp_predictor is not None and self.candidate_fp_lookup is not None:
            pred = self.fp_predictor(molecule)
            if pred is not None:
                for j in range(n_c):
                    cf = self.candidate_fp_lookup(keys[j])
                    if cf is None:
                        continue
                    inter = float(np.logical_and(pred, cf).sum())
                    union = float(pred.sum() + cf.sum() - inter)
                    cand_tani[j] = inter / union if union > 0 else 0.0
        for si, spec in enumerate(molecule.spectra):
            rows = [None] * n_c
            for j in range(n_c):
                if j not in frag_positions:
                    continue
                frag = self._fragmentor(smiles[j], spec.adduct)
                if frag is None:
                    continue
                try:
                    eif[si, j] = frag.explained_intensity_fraction(spec.mzs, spec.intensities)
                    rows[j] = candidate_row(frag, spec.mzs, spec.intensities,
                                            spec.ionization_mode,
                                            ce_bucket=spec.ce_bucket,
                                            fp_tanimoto=cand_tani[j])
                except Exception:
                    continue
            if self.scorer is not None:
                valid = [j for j, r in enumerate(rows) if r is not None]
                if len(valid) >= 2:
                    probs = self.scorer.score_set([rows[j] for j in valid])
                    for p, j in zip(probs, valid):
                        learned[si, j] = p
        return eif, learned

    def eif_scores(self, molecule: Molecule, cand_idx: np.ndarray) -> np.ndarray:
        eif, _ = self._frag_matrix(molecule, cand_idx)
        return np.nanmean(eif, axis=0)

    def learned_scores(self, molecule: Molecule, cand_idx: np.ndarray) -> np.ndarray:
        _, learned = self._frag_matrix(molecule, cand_idx)
        return np.nanmean(learned, axis=0)

    def all_channels(self, molecule: Molecule, cand_idx: np.ndarray) -> dict[str, np.ndarray]:
        eif, learned = self._frag_matrix(molecule, cand_idx)
        sim = self.sim_scores(molecule, cand_idx)
        return {
            "mass": self.mass_scores(cand_idx, molecule.neutral_mass),
            "sim": sim,
            "simclip": np.maximum(sim - 0.6, 0.0),
            "eif": np.nanmean(eif, axis=0),
            "learned": np.nanmean(learned, axis=0),
        }


CONFIGS = {
    "mass": ("mass",),
    "sim": ("sim",),
    "eif": ("eif",),
    "learned": ("learned",),
    "learned+simclip": ("learned", "simclip"),
    "learned+eif": ("learned", "eif"),
    "learned+eif+simclip": ("learned", "eif", "simclip"),
    "sim+eif+learned": ("sim", "eif", "learned"),
}


def rank_candidates(channel_scores: dict[str, np.ndarray], config: tuple[str, ...],
                    cand_idx: np.ndarray) -> np.ndarray:
    fused = np.zeros(len(cand_idx))
    for ch in config:
        fused += _zscore(channel_scores[ch])
    return cand_idx[np.argsort(-fused, kind="stable")]


def rank_gated(channel_scores: dict[str, np.ndarray], cand_idx: np.ndarray,
               sim_threshold: float = 0.9, primary: str = "learned") -> np.ndarray:
    """Class-aware gating: direct library hits rank by similarity, the rest by primary."""
    sim = channel_scores["sim"]
    if sim.size and sim.max() >= sim_threshold:
        return cand_idx[np.argsort(-sim, kind="stable")]
    return rank_candidates(channel_scores, (primary,), cand_idx)


def reciprocal_rank(ranked_keys: np.ndarray, true_key: str, k: int = 25) -> float:
    matches = np.flatnonzero(ranked_keys == true_key)
    if len(matches) == 0:
        return 0.0
    rank = int(matches[0]) + 1
    return 1.0 / rank if rank <= k else 0.0


class SimpleScorer:
    """Pickle-free reimplementation of CandidateScorer.score_set from JSON export."""

    def __init__(self, params: dict) -> None:
        self.columns = list(params["columns"])
        self.num_mean = params["num_mean"]
        self.num_std = params["num_std"]
        self.numeric_features = list(params["numeric_features"])
        self.coef = np.asarray(params["coef"], dtype=float)
        self.intercept = float(params["intercept"])

    @classmethod
    def load(cls, path) -> "SimpleScorer":
        import json

        with open(path) as f:
            return cls(json.load(f))

    def score_set(self, rows: list[dict]) -> np.ndarray:
        x = pd.DataFrame(rows)
        z = x[self.numeric_features].astype(float)
        mean = z.mean()
        std = z.std(ddof=0).replace(0.0, 1.0)
        z = (z - mean) / std
        for col in self.numeric_features:
            z[col] = (z[col] - self.num_mean[col]) / self.num_std[col]
        z = z.reindex(columns=self.columns, fill_value=0.0).to_numpy(dtype=float)
        p_decoy = 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))
        return 1.0 - p_decoy
