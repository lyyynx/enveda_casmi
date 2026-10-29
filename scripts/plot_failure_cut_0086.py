"""Two self-contained failure figures for cut_0086:

1. plots/failure_cut_0086_truth.png -- the TRUE structure, its spectrum peaks
   explained by in-silico fragments, and the matched fragment structures.
2. plots/failure_cut_0086_pred.png -- the same layout for our top-1 (wrong)
   assignment.
"""

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib import gridspec
from rdkit import Chem
from rdkit.Chem import Draw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enveda_casmi26.cleaning import CleanConfig, SpectrumCleaner
from enveda_casmi26.fragmentation import Fragmentor

MID = "cut_0086"
TRUTH_SMI = "COc1ccccc1CC1CCCN1C(=O)CCc1nc(C)no1"
TOP1_SMI = "COc1cccc(COCC(=O)N2CCCC(c3ccn[nH]3)C2)c1"
COLOR = {"truth": "#2ca02c", "pred": "#d62728"}

test = pq.read_table(PROJECT_ROOT / "data" / "test_cut.parquet").to_pandas()
g = test[test.molecule_id == MID]
cleaner = SpectrumCleaner(CleanConfig())
r = g.iloc[-1]
res = cleaner.clean(np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float),
                    precursor_mz=float(r.precursor_mz))
mzs, ints = res.mzs, res.intensities


def matches_for(smiles):
    frag = Fragmentor(smiles, adduct="[M+H]+", tolerance_ppm=10.0)
    m = frag.match_peaks(mzs, ints)
    m = m.drop_duplicates("peak_index")  # best match per peak
    mask = np.zeros(len(mzs), dtype=bool)
    mask[m.peak_index.to_numpy(int)] = True
    return m, mask


def mol_image(mol_list, legends):
    img = Draw.MolsToGridImage(mol_list, molsPerRow=len(mol_list) if len(mol_list) <= 5 else 5,
                               subImgSize=(300, 220), legends=legends, useSVG=False)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return plt.imread(buf)


def figure(smiles, tag, title_note):
    m, mask = matches_for(smiles)
    frag_rows = m[m.fragment_smiles.notna() & (m.fragment_smiles != "")]
    frag_rows = frag_rows.drop_duplicates("fragment_smiles")
    frag_rows = frag_rows.sort_values("intensity", ascending=False) if "intensity" in frag_rows else frag_rows

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.1, 1.15, 1.5], hspace=0.35)

    # -- structure
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    main = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(main, size=(1100, 340))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    ax0.imshow(plt.imread(buf))
    ax0.set_title(title_note, fontsize=13, pad=2)

    # -- spectrum
    ax1 = fig.add_subplot(gs[1])
    ax1.vlines(mzs, 0, ints, color="0.78", lw=1.3)
    ax1.vlines(mzs[mask], 0, ints[mask], color=COLOR[tag], lw=2.0,
               label=f"explained by this structure: {mask.sum()}/{len(mzs)} peaks, "
                     f"{ints[mask].sum() / ints.sum():.0%} of intensity")
    for i in np.where(mask)[0]:
        ax1.annotate(f"{mzs[i]:.2f}", (mzs[i], ints[i] + 0.02), ha="center", fontsize=7.5,
                     color=COLOR[tag])
    ax1.set_xlabel("m/z")
    ax1.set_ylabel("normalized intensity")
    ax1.set_ylim(0, 1.18)
    ax1.legend(fontsize=9, loc="upper right")

    # -- fragments
    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    ax2.set_title("matched in-silico fragments (observed m/z, error ppm)", fontsize=11)
    if len(frag_rows):
        mols, legends = [], []
        for _, fr in frag_rows.head(10).iterrows():
            fm = Chem.MolFromSmiles(fr.fragment_smiles)
            if fm is None:
                continue
            mols.append(fm)
            legends.append(f"m/z {fr.peak_mz:.4f} ({fr.error_ppm:+.1f} ppm, {fr.kind})")
        if mols:
            ax2.imshow(mol_image(mols, legends))
        else:
            ax2.text(0.5, 0.5, "no drawable fragments", ha="center")
    return fig


fig = figure(TRUTH_SMI, "truth",
             f"{MID} -- TRUE structure: COc1ccccc1CC1CCCN1C(=O)CCc1nc(C)no1\n"
             "present in candidate pool at +1.46 ppm, but dropped by the 200-candidate cap "
             "(position 236 of 410 same-window candidates)")
fig.savefig(PROJECT_ROOT / "plots" / "failure_cut_0086_truth.png", dpi=160, bbox_inches="tight")
plt.close(fig)

fig = figure(TOP1_SMI, "pred",
             f"{MID} -- our TOP-1 (wrong) assignment: COc1cccc(COCC(=O)N2CCCC(c3ccn[nH]3)C2)c1\n"
             "a same-mass isomer that outranked the truth via analog library similarity")
fig.savefig(PROJECT_ROOT / "plots" / "failure_cut_0086_pred.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("saved plots/failure_cut_0086_truth.png")
print("saved plots/failure_cut_0086_pred.png")
