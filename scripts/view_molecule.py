"""View MS2 spectra for a selected molecule.

Usage:
    uv run python scripts/view_molecule.py m_005e53
    uv run python scripts/view_molecule.py --source train <INCHIKEY14_or_SMILES>
    uv run python scripts/view_molecule.py m_005e53 --meta adduct,precursor_mz,instrument_type --save
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"

DEFAULT_META = "adduct,precursor_mz,collision_energy_orig,instrument_type"


def load_data(source: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{source}.parquet")


def select_spectra(df: pd.DataFrame, molecule: str, source: str) -> pd.DataFrame:
    """Select all spectra belonging to one molecule/structure."""
    if source == "test":
        return df[df["molecule_id"] == molecule].reset_index(drop=True)

    # train: allow selection by (partial) inchikey or SMILES
    mask = df["inchikey14"].str.startswith(molecule) | df["inchikey"].str.startswith(molecule)
    if molecule.lower().startswith(("c", "n", "o")):
        mask |= df["normalized_smiles"].str.startswith(molecule, na=False)
    return df[mask].reset_index(drop=True)


def plot_spectrum(ax, mzs, intensities, title: str, meta_text: str) -> None:
    ax.vlines(mzs, 0, intensities, color="tab:blue", linewidth=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("m/z")
    ax.set_ylabel("rel. intensity")
    ax.set_ylim(0, 1.05)
    if meta_text:
        ax.text(
            0.02, 0.98, meta_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=7, color="tab:gray",
        )


def meta_str(row: pd.Series, meta_cols: list[str]) -> str:
    parts = []
    for col in meta_cols:
        if col not in row or pd.isna(row[col]):
            continue
        val = row[col]
        if isinstance(val, np.ndarray):
            val = "/".join(f"{v:g}" for v in val[:3])
        elif isinstance(val, float):
            val = f"{val:.4g}"
        parts.append(f"{col}: {val}")
    return "\n".join(parts)


def view_molecule(
    molecule: str,
    source: str = "test",
    meta_cols: list[str] | None = None,
    max_spectra: int = 12,
    save: bool = False,
    show: bool = True,
) -> plt.Figure:
    meta_cols = [c.strip() for c in (meta_cols or DEFAULT_META.split(",")) if c.strip()]

    df = load_data(source)
    rows = select_spectra(df, molecule, source)

    if rows.empty:
        available = df["molecule_id"].unique()[:10] if source == "test" else df["inchikey14"].unique()[:10]
        raise SystemExit(
            f"No spectra found for '{molecule}' in {source}.parquet.\n"
            f"Examples to try: {', '.join(map(str, available))}"
        )

    print(f"{len(rows)} spectra found for '{molecule}' in {source}.parquet\n")
    if "num_peaks" not in rows.columns:
        rows["num_peaks"] = [len(m) for m in rows["ms2_mzs"]]
    id_cols = ["molecule_id", "spectrum_id"] if source == "test" else ["inchikey14", "normalized_smiles"]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(rows[id_cols + [c for c in meta_cols if c in rows.columns] + ["num_peaks"]])

    n = min(len(rows), max_spectra)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"Molecule {molecule} ({source} set)", fontsize=12)

    for i in range(n):
        row = rows.iloc[i]
        ax = axes[i // ncols][i % ncols]
        title = f"{row.get('spectrum_id', f'row {i}')} — {row['num_peaks']} peaks"
        plot_spectrum(ax, row["ms2_mzs"], row["ms2_normalized_intensities"], title, meta_str(row, meta_cols))

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if n < len(rows):
        fig.text(0.5, 0.01, f"Showing {n} of {len(rows)} spectra", ha="center", fontsize=8, color="gray")

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    if save:
        PLOTS_DIR.mkdir(exist_ok=True)
        out = PLOTS_DIR / f"spectra_{source}_{molecule.replace('/', '_')}.png"
        fig.savefig(out, dpi=150)
        print(f"\nSaved plot to {out}")
    if show:
        plt.show()
    return fig


def main() -> None:
    @dataclass
    class Argsy:
        molecule: str
        meta: list[str]
        source: str = "train"
        max_spectra: int = 12
        save: bool = False

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("molecule", help="molecule_id (test) or inchikey/SMILES prefix (train)")
    parser.add_argument("--source", choices=["test", "train"], default="test")
    parser.add_argument("--meta", default=DEFAULT_META, help="comma-separated metadata columns to annotate")
    parser.add_argument("--max-spectra", type=int, default=12, help="max number of spectra to plot")
    parser.add_argument("--save", action="store_true", help="save plot to plots/ instead of only showing")
    #args = parser.parse_args()
    args = Argsy("AIJULSRZWUXGPQ", meta=DEFAULT_META, source="train", save=True)

    view_molecule(
        molecule=args.molecule,
        source=args.source,
        meta_cols=args.meta.split(","),
        max_spectra=args.max_spectra,
        save=args.save,
    )


if __name__ == "__main__":
    main()
