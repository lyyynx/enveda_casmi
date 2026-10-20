"""Explain a spectrum: which peak comes from which fragment, with which loss.

Given a spectrum (train/test row or molecule) and a candidate structure,
annotate every peak with the in-silico fragment that explains it: cleavage
type, neutral loss, H-shift, mass error, and fragment identity.

  --discovery  additionally try the permissive discovery config (all losses,
               serial loss chains, h-shifts +/-3) on peaks the default
               fragmenter leaves unexplained. Those matches are reported
               separately as "speculative".
  --plot       writes a three-panel report figure: annotated spectrum,
               fragment structures, and a parent-structure map showing
               where each fragment sits inside the molecule.

Usage:
    uv run python scripts/explain_spectrum.py --source test --molecule-id m_ff7740 \
        --discovery --plot report.png
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from enveda_casmi26.fragmentation import LOSS_DEFS, Fragmentor

DATA_DIR = PROJECT_ROOT / "data"

KIND_LABELS = {
    "side_chain": "side-chain cleavage (1 cut, acyclic bond)",
    "ring_fission": "ring fission (2 cuts in one ring)",
    "precursor": "intact precursor with neutral loss",
}


def fragment_formula(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcMolFormula(mol) if mol else "?"


def annotate(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    m["formula"] = m["fragment_smiles"].map(fragment_formula)
    m["rel_int_pct"] = 100.0 * m["intensity"]
    return m


def print_table(m: pd.DataFrame, top: int, title: str) -> None:
    cols = ["peak_mz", "rel_int_pct", "kind", "neutral_loss", "h_shift", "error_ppm",
            "formula", "frag_heavy_atoms", "fragment_smiles"]
    table = m[cols].copy()
    table["rel_int_pct"] = table["rel_int_pct"].round(1)
    table = table.sort_values("rel_int_pct", ascending=False).head(top)
    print(f"\n{title} (top {len(table)}, by intensity):")
    with pd.option_context("display.width", 220, "display.max_colwidth", 62):
        print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["train", "test"], default="test")
    parser.add_argument("--molecule-id", default=None)
    parser.add_argument("--molecule-row", type=int, default=None)
    parser.add_argument("--smiles", default=None, help="candidate structure to explain with")
    parser.add_argument("--spectrum-index", type=int, default=None,
                        help="which spectrum of the molecule (default: most peaks)")
    parser.add_argument("--tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=30, help="peaks to show in tables")
    parser.add_argument("--discovery", action="store_true",
                        help="also explain unexplained peaks with the permissive config")
    parser.add_argument("--plot", default=None, help="save the report figure to this path")
    args = parser.parse_args()

    df = pd.read_parquet(DATA_DIR / f"{args.source}_clean.parquet")
    if args.molecule_id:
        sub = df[df["molecule_id"] == args.molecule_id]
        if sub.empty:
            raise SystemExit(f"molecule_id {args.molecule_id!r} not found")
        if args.spectrum_index is not None and args.spectrum_index < len(sub):
            row = sub.iloc[args.spectrum_index]
        else:
            row = (sub.nlargest(1, "num_peaks_clean").iloc[0]
                   if "num_peaks_clean" in sub else sub.iloc[0])
        n_spectra = len(sub)
    else:
        if args.molecule_row is None:
            raise SystemExit("specify --molecule-id or --molecule-row")
        row = df.iloc[args.molecule_row]
        n_spectra = 1

    if args.smiles:
        smiles, origin = args.smiles, "user-supplied"
    elif args.source == "train":
        smiles, origin = row["normalized_smiles"], "train label"
    else:
        hits = PROJECT_ROOT / "benchmark" / "test_similarity_hits.csv"
        if hits.exists():
            h = pd.read_csv(hits)
            h = h[(h["molecule_id"] == row["molecule_id"]) & (h["rank"] == 1)]
            if not h.empty:
                smiles, origin = h["normalized_smiles"].iloc[0], "best library hit"
        else:
            raise SystemExit("test spectra need --smiles or a similarity-hits file")

    mzs = np.asarray(row["ms2_mzs"], dtype=float)
    ints = np.asarray(row["ms2_normalized_intensities"], dtype=float)

    frag = Fragmentor(smiles, adduct=row["adduct"], tolerance_ppm=args.tolerance_ppm)
    matches = frag.match_peaks(mzs, ints)
    if matches.empty:
        raise SystemExit("no peaks explained by this structure")
    m = annotate(matches)

    spec_m = None
    if args.discovery:
        dfrag = Fragmentor(smiles, adduct=row["adduct"], tolerance_ppm=args.tolerance_ppm,
                           include_precursor_losses=False, max_serial_losses=3,
                           neutral_losses=tuple(LOSS_DEFS),
                           h_shifts=(-3, -2, -1, 0, 1, 2, 3))
        unexp = [i for i in range(len(mzs)) if i not in set(matches["peak_index"])]
        if unexp:
            d = dfrag.match_peaks(mzs[unexp], ints[unexp])
            if not d.empty:
                d["peak_index"] = [unexp[i] for i in d["peak_index"]]
                d["intensity"] = ints[d["peak_index"].to_numpy()]
                spec_m = annotate(d)

    spec_id = row["molecule_id"] if "molecule_id" in row.index else row["inchikey14"]
    print("=" * 100)
    print(f"spectrum      : {spec_id} (spectrum 1 of {n_spectra})")
    print(f"adduct/mode   : {row['adduct']} / {row['ionization_mode']}")
    print(f"precursor m/z : {row['precursor_mz']:.4f} (theoretical {frag.precursor_mz:.4f})")
    print(f"candidate     : {smiles}   [{origin}]")
    print("=" * 100)

    total_int = float(ints.sum())
    explained_int = m["intensity"].sum() / total_int
    spec_int = spec_m["intensity"].sum() / total_int if spec_m is not None else 0.0
    print(f"\nexplained intensity: {explained_int:.1%} (default config)"
          + (f"   + {spec_int:.1%} speculative (discovery config)   "
             f"total {(explained_int + spec_int):.1%}" if spec_m is not None else ""))
    print(f"peaks explained: {len(m)}/{len(mzs)} default"
          + (f", +{len(spec_m)} speculative" if spec_m is not None else ""))

    print_table(m, args.top, "explained peaks")
    if spec_m is not None:
        print_table(spec_m, args.top, "speculative matches (permissive config)")

    still_unexplained = len(mzs) - len(set(matches["peak_index"])
                                      | (set(spec_m["peak_index"]) if spec_m is not None else set()))
    order = [i for i in range(len(mzs))
             if i not in set(matches["peak_index"])
             | (set(spec_m["peak_index"]) if spec_m is not None else set())]
    order.sort(key=lambda i: -ints[i])
    if order:
        print(f"\nstill unexplained: {len(order)} "
              f"({100 * sum(ints[i] for i in order) / total_int:.1f}% of intensity). "
              f"Most intense:")
        for i in order[:8]:
            print(f"  m/z {mzs[i]:.4f}  ({100 * ints[i]:.1f}%)")

    for label, frame in (("explained", m), ("speculative", spec_m)):
        if frame is None:
            continue
        kind_int = frame.groupby("kind")["intensity"].sum() / total_int
        print(f"\n{label} intensity by cleavage type:")
        for k, v in kind_int.sort_values(ascending=False).items():
            print(f"  {k:<14} {v:>6.1%}   ({KIND_LABELS.get(k, k)})")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from rdkit.Chem import Draw

        kind_colors = {"side_chain": "tab:green", "ring_fission": "tab:orange",
                       "precursor": "tab:blue"}
        matched = m.set_index("peak_index")
        spec_matched = spec_m.set_index("peak_index") if spec_m is not None else None

        fig = plt.figure(figsize=(17, 15))
        gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.1, 1.1], hspace=0.16)
        ax = fig.add_subplot(gs[0])
        for i, (mz, it) in enumerate(zip(mzs, ints)):
            if i in matched.index:
                color = kind_colors.get(matched.loc[i, "kind"], "tab:green")
            elif spec_matched is not None and i in spec_matched.index:
                color = "tab:purple"
            else:
                color = "0.78"
            ax.plot([mz, mz], [0, it], color=color, lw=1.5)

        for _, r in m.nlargest(12, "intensity").iterrows():
            label = r["formula"]
            if r["neutral_loss"] != "none":
                label += f" −{r['neutral_loss']}"
            label += f" ({r['error_ppm']:+.0f}ppm)"
            ax.annotate(label, (r["peak_mz"], r["intensity"]), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7,
                        rotation=60, color="0.25")
        if spec_matched is not None:
            for _, r in spec_m.nlargest(10, "intensity").iterrows():
                label = r["formula"]
                if r["neutral_loss"] != "none":
                    label += f" −{r['neutral_loss']}"
                ax.annotate(f"[spec] {label}", (r["peak_mz"], r["intensity"]),
                            xytext=(0, 7), textcoords="offset points", ha="center",
                            fontsize=6.5, rotation=60, color="tab:purple")
        handles = [Line2D([0], [0], color=c, lw=3)
                   for c in list(kind_colors.values()) + ["tab:purple", "0.78"]]
        ax.legend(handles, ["side-chain cleavage", "ring fission", "precursor loss",
                            "speculative (discovery config)", "unexplained"],
                  loc="upper left", fontsize=8)
        ax.set_ylabel("relative intensity")
        ax.set_title(f"{spec_id}  —  {smiles}\n"
                     f"explained {explained_int:.1%}"
                     + (f" (+{spec_int:.1%} speculative)" if spec_m is not None else "")
                     + f", {len(m)}/{len(mzs)} peaks  ({row['adduct']}, "
                       f"precursor {row['precursor_mz']:.4f})", fontsize=10)

        ax.axvline(row["precursor_mz"], color="0.4", ls="--", lw=0.9)
        xmin, xmax = ax.get_xlim()
        frac = (row["precursor_mz"] - xmin) / (xmax - xmin)
        w, h = 0.21, 0.40
        x0 = min(max(frac - w - 0.02, 0.02), 0.98 - w)
        axins = ax.inset_axes([x0, 1.0 - h - 0.03, w, h])
        axins.axis("off")
        axins.imshow(np.asarray(Draw.MolToImage(frag.mol, size=(420, 420))))
        axins.set_title("full molecule (precursor)", fontsize=8, color="0.25")

        m_marked = m.assign(spec=False)
        if spec_m is not None:
            m_marked = pd.concat([m_marked, spec_m.assign(spec=True)])
        top_frags = m_marked.nlargest(12, "intensity").drop_duplicates("fragment_smiles")
        ax2 = fig.add_subplot(gs[1])
        ax2.axis("off")
        mols, legends = [], []
        for _, r in top_frags.iterrows():
            mol = Chem.MolFromSmiles(r["fragment_smiles"])
            if mol is None:
                continue
            mols.append(mol)
            loss = f"−{r['neutral_loss']}" if r["neutral_loss"] != "none" else ""
            tag = " [spec]" if r["spec"] else ""
            legends.append(f"m/z {r['peak_mz']:.3f}  {r['formula']}{loss}  "
                           f"{r['rel_int_pct']:.0f}%{tag}")
        if mols:
            ax2.imshow(np.asarray(Draw.MolsToGridImage(
                mols, legends=legends, molsPerRow=6, subImgSize=(340, 260), useSVG=False)))
            ax2.set_title("fragment structures", fontsize=9)

        ax3 = fig.add_subplot(gs[2])
        ax3.axis("off")
        hl_mols, hl_atoms, hl_bonds, hl_legends = [], [], [], []
        for _, r in top_frags.iterrows():
            side = list(r["side_atoms"])
            cut = ast.literal_eval(r["cut_bonds"]) if r["cut_bonds"] else ()
            bond_ids = [frag.mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in cut]
            hl_mols.append(frag.mol)
            hl_atoms.append(side)
            hl_bonds.append(bond_ids)
            loss = f" −{r['neutral_loss']}" if r["neutral_loss"] != "none" else ""
            hl_legends.append(f"m/z {r['peak_mz']:.2f}{loss}  "
                              f"({r['kind']}, {r['rel_int_pct']:.0f}%)"
                              + (" [spec]" if r["spec"] else ""))
        if hl_mols:
            ax3.imshow(np.asarray(Draw.MolsToGridImage(
                hl_mols, legends=hl_legends, highlightAtomLists=hl_atoms,
                highlightBondLists=hl_bonds, molsPerRow=6, subImgSize=(400, 320),
                useSVG=False)))
            ax3.set_title("parent-structure map: highlighted atoms form the fragment, "
                          "highlighted bonds are cut", fontsize=9)

        fig.savefig(args.plot, dpi=140, bbox_inches="tight")
        print(f"\nreport figure saved to {args.plot}")


if __name__ == "__main__":
    main()
