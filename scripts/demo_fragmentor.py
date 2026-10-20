"""Run the Fragmentor on real training spectra and report/plot peak matches.

Usage:
    uv run python scripts/demo_fragmentor.py            # aspirin if present, else random molecule
    uv run python scripts/demo_fragmentor.py <SMILES>   # pick a specific structure
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from enveda_casmi26.fragmentation import Fragmentor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def pick_example(df: pd.DataFrame, smiles: str | None) -> pd.Series:
    if smiles:
        hits = df[df["normalized_smiles"] == smiles]
    else:
        hits = df[df["normalized_smiles"] == ASPIRIN]
    if hits.empty:
        hits = df[
            (df["adduct"] == "[M+H]+")
            & df["num_peaks"].between(20, 80)
            & df["instrument_type"].notna()
        ]
        hits = hits.sample(1, random_state=42)
    return hits.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smiles", nargs="?", default=None, help="normalized SMILES to analyze")
    parser.add_argument("--tolerance-ppm", type=float, default=10.0)
    args = parser.parse_args()

    cols = [
        "normalized_smiles", "adduct", "num_peaks", "precursor_mz",
        "ms2_mzs", "ms2_normalized_intensities", "ionization_mode", "instrument_type",
        "collision_energy_orig",
    ]
    print("Loading training data ...")
    df = pd.read_parquet(DATA_DIR / "train.parquet", columns=cols)
    row = pick_example(df, args.smiles)

    smiles = row["normalized_smiles"]
    print(f"\nSMILES:        {smiles}")
    print(f"Adduct:        {row['adduct']}   mode: {row['ionization_mode']}")
    print(f"Instrument:    {row['instrument_type']}   CE: {row['collision_energy_orig']}")
    print(f"Peaks:         {row['num_peaks']}")

    frag = Fragmentor(smiles, adduct=row["adduct"], tolerance_ppm=args.tolerance_ppm)
    precursor_ppm = (frag.precursor_mz - row["precursor_mz"]) / row["precursor_mz"] * 1e6
    print(f"Neutral mass: {frag.neutral_mass:.4f}   theoretical precursor m/z: {frag.precursor_mz:.4f}"
          f"   (reported: {row['precursor_mz']:.4f}, {precursor_ppm:+.1f} ppm)")
    print(f"Theoretical fragment candidates: {len(frag.enumerate_fragments())}")

    matches = frag.match_peaks(row["ms2_mzs"], row["ms2_normalized_intensities"])
    n_matched = len(matches)
    explained = frag.explained_intensity_fraction(row["ms2_mzs"], row["ms2_normalized_intensities"])
    print(f"\nMatched {n_matched}/{row['num_peaks']} peaks within {args.tolerance_ppm} ppm"
          f" ({explained:.1%} of total intensity)")

    if not matches.empty:
        with pd.option_context("display.max_columns", None, "display.width", 250):
            print(matches.sort_values("intensity", ascending=False).head(15))

    fig, ax = plt.subplots(figsize=(14, 5))
    matched_mzs = set(matches["peak_mz"]) if not matches.empty else set()
    for m, inten in zip(row["ms2_mzs"], row["ms2_normalized_intensities"]):
        color = "tab:orange" if any(abs(m - mm) < 1e-6 for mm in matched_mzs) else "tab:blue"
        ax.vlines(m, 0, inten, color=color, linewidth=1.0)
    ax.set_xlabel("m/z")
    ax.set_ylabel("rel. intensity")
    ax.set_title(
        f"{smiles}  |  {row['adduct']}  |  matched {n_matched}/{row['num_peaks']} peaks,"
        f" {explained:.0%} intensity  (orange = matched)"
    )
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / f"fragmentor_{smiles[:30].replace('/', '_')}.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
