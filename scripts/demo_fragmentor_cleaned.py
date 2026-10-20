"""Run the Fragmentor on a cleaned training spectrum to verify compatibility.

Usage:
    uv run python scripts/demo_fragmentor_cleaned.py
"""

from pathlib import Path

import pandas as pd

from enveda_casmi26.cleaning import CleanConfig, SpectrumCleaner
from enveda_casmi26.fragmentation import Fragmentor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def main() -> None:
    print("Loading raw training data ...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")
    hits = df[df["normalized_smiles"] == ASPIRIN]
    if hits.empty:
        hits = df[(df["adduct"] == "[M+H]+") & df["num_peaks"].between(20, 80)]
        hits = hits.sample(1, random_state=42)
    row = hits.iloc[0]
    smiles = row["normalized_smiles"]

    cleaner = SpectrumCleaner(CleanConfig())
    clean = cleaner.clean(row["ms2_mzs"], row["ms2_normalized_intensities"], precursor_mz=row["precursor_mz"])
    print(f"SMILES: {smiles}   adduct: {row['adduct']}")
    print(f"raw: {clean.n_input} peaks -> cleaned: {clean.n_kept} peaks, flag: {clean.drop_reason or 'none'}")
    print(f"removed by step: {clean.removed_by_step}")

    frag = Fragmentor(smiles, adduct=row["adduct"], tolerance_ppm=10.0)
    raw_expl = frag.explained_intensity_fraction(row["ms2_mzs"], row["ms2_normalized_intensities"])
    clean_expl = frag.explained_intensity_fraction(clean.mzs, clean.intensities)
    print(f"\nFragmentor explained intensity: raw {raw_expl:.1%} -> cleaned {clean_expl:.1%}")

    matches = frag.match_peaks(clean.mzs, clean.intensities)
    print(f"matched {len(matches)}/{clean.n_kept} cleaned peaks within 10 ppm")
    if not matches.empty:
        with pd.option_context("display.max_columns", None, "display.width", 250):
            print(matches.sort_values("intensity", ascending=False).head(10))


if __name__ == "__main__":
    main()
