"""MS2 spectrum curation.

Applies common cleaning steps to raw spectra:
  1. drop peaks above precursor m/z (+ margin),
  2. relative-intensity floor (relative to base peak),
  3. deisotoping (removes C-13 companions ~1.003 Da above a monoisotopic peak),
  4. mass-defect junk filter (no organic z=1 ion has m/z fractional part in
     the mid-decimal valley below ~350 Da; such peaks are ghosts/noise),
  5. top-K most intense peaks per m/z window and/or a global top-N cap,
  6. square-root / log intensity transform,
  7. renormalization to base peak = 1.

Spectra that fall below ``min_peaks`` after cleaning are flagged
(``drop_reason="too_few_peaks"``) but never silently removed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

ISOTOPE_STEP = 1.0033548


@dataclass(frozen=True)
class CleanConfig:
    relative_intensity_floor: float = 0.001
    precursor_margin_da: float | None = 1.5
    deisotope: bool = True
    isotope_tolerance_da: float = 0.01
    isotope_max_ratio: float = 1.0
    isotope_max_count: int = 3
    mass_defect_filter: bool = True
    mass_defect_lo: float = 0.44
    mass_defect_hi: float = 0.56
    mass_defect_max_mz: float = 350.0
    top_k_per_window: int | None = 6
    window_da: float = 50.0
    max_peaks: int | None = 128
    min_peaks: int = 5
    intensity_transform: str = "sqrt"
    renormalize: bool = True


@dataclass
class CleanResult:
    mzs: np.ndarray
    intensities: np.ndarray
    n_input: int
    n_kept: int
    drop_reason: str | None
    removed_by_step: dict[str, int]

    @property
    def dropped(self) -> bool:
        return self.drop_reason is not None


class SpectrumCleaner:
    """Clean single spectra or whole DataFrames according to a CleanConfig."""

    def __init__(self, config: CleanConfig | None = None) -> None:
        if config is None:
            config = CleanConfig()
        if config.intensity_transform not in ("sqrt", "log", "none"):
            raise ValueError(f"Unknown transform {config.intensity_transform!r}")
        self.config = config

    def clean(self, mzs, intensities, precursor_mz: float | None = None) -> CleanResult:
        cfg = self.config
        mzs = np.asarray(mzs, dtype=float)
        ints = np.asarray(intensities, dtype=float)
        if len(mzs) != len(ints):
            raise ValueError(f"mzs ({len(mzs)}) and intensities ({len(ints)}) length mismatch")
        n_input = len(mzs)
        removed: Counter[str] = Counter()

        order = np.argsort(mzs, kind="stable")
        mzs, ints = mzs[order], ints[order]

        if len(mzs) == 0:
            return CleanResult(mzs, ints, 0, 0, "empty", dict(removed))

        if cfg.precursor_margin_da is not None and precursor_mz is not None and np.isfinite(precursor_mz):
            mask = mzs <= precursor_mz + cfg.precursor_margin_da
            removed["above_precursor"] = int((~mask).sum())
            mzs, ints = mzs[mask], ints[mask]

        if len(mzs) and cfg.relative_intensity_floor > 0:
            floor = cfg.relative_intensity_floor * max(float(ints.max()), 1e-12)
            mask = ints >= floor
            removed["below_floor"] = int((~mask).sum())
            mzs, ints = mzs[mask], ints[mask]

        if cfg.deisotope and len(mzs) > 1:
            mzs, ints, n_iso = self._deisotope(mzs, ints)
            removed["isotope_companions"] = n_iso

        if cfg.mass_defect_filter and len(mzs):
            frac = np.abs(mzs - np.round(mzs))
            mask = ~((frac >= cfg.mass_defect_lo) & (frac <= cfg.mass_defect_hi)
                     & (mzs < cfg.mass_defect_max_mz))
            removed["mass_defect_junk"] = int((~mask).sum())
            mzs, ints = mzs[mask], ints[mask]

        if len(mzs) and cfg.top_k_per_window is not None:
            mzs, ints, n_cut = self._top_k_per_window(mzs, ints)
            removed["window_top_k"] = n_cut

        if len(mzs) and cfg.max_peaks is not None and len(mzs) > cfg.max_peaks:
            keep = np.argsort(-ints, kind="stable")[: cfg.max_peaks]
            removed["global_cap"] = len(mzs) - cfg.max_peaks
            mzs, ints = mzs[keep], ints[keep]
            order = np.argsort(mzs, kind="stable")
            mzs, ints = mzs[order], ints[order]

        n_kept = len(mzs)
        if n_kept == 0:
            drop_reason = "empty"
        elif n_kept < cfg.min_peaks:
            drop_reason = "too_few_peaks"
        else:
            drop_reason = None

        if n_kept and cfg.intensity_transform != "none":
            if cfg.intensity_transform == "sqrt":
                ints = np.sqrt(ints)
            else:
                ints = np.log10(1.0 + 9.0 * ints)
        if n_kept and cfg.renormalize:
            ints = ints / ints.max()

        return CleanResult(mzs, ints, n_input, n_kept, drop_reason, dict(removed))

    def _deisotope(self, mzs: np.ndarray, ints: np.ndarray):
        cfg = self.config
        keep = np.ones(len(mzs), dtype=bool)
        for k in range(1, cfg.isotope_max_count + 1):
            target = k * ISOTOPE_STEP
            lo = np.searchsorted(mzs, mzs + target - cfg.isotope_tolerance_da)
            hi = np.searchsorted(mzs, mzs + target + cfg.isotope_tolerance_da)
            for i in range(len(mzs)):
                if not keep[i]:
                    continue
                for j in range(lo[i], hi[i]):
                    if keep[j] and ints[j] <= ints[i] * cfg.isotope_max_ratio:
                        keep[j] = False
        removed = int((~keep).sum())
        return mzs[keep], ints[keep], removed

    def _top_k_per_window(self, mzs: np.ndarray, ints: np.ndarray):
        cfg = self.config
        windows = np.floor_divide(mzs, cfg.window_da).astype(np.int64)
        n_windows = int(windows.max() + 1)
        counts = np.zeros(n_windows + 1, dtype=np.int64)
        keep = np.zeros(len(mzs), dtype=bool)
        for idx in np.argsort(-ints, kind="stable"):
            w = int(windows[idx])
            if counts[w] < cfg.top_k_per_window:
                keep[idx] = True
                counts[w] += 1
        removed = int((~keep).sum())
        mzs, ints = mzs[keep], ints[keep]
        return mzs, ints, removed

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        mzs_col: str = "ms2_mzs",
        int_col: str = "ms2_normalized_intensities",
        precursor_col: str | None = "precursor_mz",
        progress_every: int | None = 250_000,
    ) -> pd.DataFrame:
        """Return a copy of df with cleaned spectra columns and cleaning metadata."""
        drop_cols = [c for c in (mzs_col, int_col) if c in df.columns]
        out = df.drop(columns=drop_cols).copy()

        cleaned_mzs, cleaned_ints = [], []
        n_peaks_clean, n_removed_total, drop_reasons = [], [], []
        prec = df[precursor_col].to_numpy() if precursor_col and precursor_col in df.columns else [None] * len(df)
        mzs_series = df[mzs_col].tolist()
        ints_series = df[int_col].tolist()

        for i, (mzs, ints) in enumerate(zip(mzs_series, ints_series)):
            res = self.clean(mzs, ints, precursor_mz=prec[i])
            cleaned_mzs.append(res.mzs)
            cleaned_ints.append(res.intensities)
            n_peaks_clean.append(res.n_kept)
            n_removed_total.append(res.n_input - res.n_kept)
            drop_reasons.append(res.drop_reason or "")
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  cleaned {i + 1:,}/{len(df):,} spectra")

        out[mzs_col] = cleaned_mzs
        out[int_col] = cleaned_ints
        out["num_peaks_clean"] = n_peaks_clean
        out["n_removed"] = n_removed_total
        out["drop_reason"] = drop_reasons
        return out

    @staticmethod
    def summarize(df: pd.DataFrame, label: str) -> None:
        print(f"\n===== Cleaning summary: {label} =====")
        print(f"  spectra: {len(df):,}")
        flagged = df["drop_reason"].replace("", pd.NA).value_counts(dropna=False)
        print(f"  flagged: {flagged.to_dict()}")
        npc = df["num_peaks_clean"]
        print(
            f"  peaks/spectrum after cleaning: min={npc.min()} median={npc.median():.0f}"
            f" p95={npc.quantile(0.95):.0f} max={npc.max()}"
        )
        print(f"  peaks removed total: {df['n_removed'].sum():,}")


def quick_config(**overrides) -> CleanConfig:
    return replace(CleanConfig(), **overrides)
