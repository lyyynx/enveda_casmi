# enveda-casmi26

Solution code for the Enveda CASMI26 molecule identification from mass spectra competition.

## Setup

```bash
uv sync
```

## Downloading and cleaning the data

The competition data is not included in this repository (it is too large).
Download it with the helper script, which uses [kagglehub](https://pypi.org/project/kagglehub/)
to fetch the Kaggle competition dataset `enveda-CASMI26-molecule-id-mass-spectra`
into the `data/` folder:

```bash
uv run python scripts/download_data.py
```

You will need Kaggle API credentials (`~/.kaggle/kaggle.json`) for the download to work.

The raw spectra need cleaning before they are useful: the raw peak lists contain
noise peaks, isotope peaks, and peaks above the precursor (physically impossible
in MS2). `scripts/clean_data.py` applies our cleaning pipeline to both files and
writes `train_clean.parquet` / `test_clean.parquet`. With the defaults we used it

- removes peaks below a relative intensity floor of 0.001,
- removes peaks above precursor m/z + 1.5 Da,
- deisotopes (0.01 Da tolerance, up to 3 trailing isotope peaks),
- keeps only the top 6 peaks per 50 Da mass window (max 128 peaks total),
- sqrt-transforms intensities and renormalizes.

```bash
uv run python scripts/clean_data.py            # full run; --sample 10000 for a quick trial
```

The same cleaning is applied on the fly at runtime by the kernel
(`enveda_casmi26.cleaning.SpectrumCleaner`), so the cleaned parquets are mainly
used for local analysis and validation.

## Local validation split

`scripts/make_split.py` carves a leak-free validation split out of the training
data by holding out entire structures (all their spectra) and producing
`train_cut.parquet` (drop-in replacement for `train.parquet`), `test_cut.parquet`
(mirrors the test schema + truth columns), `pool_extra.parquet` /
`val_external_pool.parquet` (held-out structures as external-DB candidates) and
`cut_map.csv`:

```bash
uv run python scripts/make_split.py            # 400 normal + 60 defect molecules
```

The training labels also contain systematic precursor/adduct errors (isotope
misassignments, adduct swaps, in-source losses, dimers). Repair them with
`scripts/repair_precursor_labels.py`, which corrects or drops flagged rows and
never modifies the source file (see its docstring for the exact rules):

```bash
uv run python scripts/repair_precursor_labels.py
```

## Derived pool assets

The remaining `data/` files and kernel assets are produced by these scripts, in
this order:

```bash
uv run python scripts/build_coconut_pool.py     # COCONUT external candidate pool (from data/coconut/)
uv run python scripts/build_exact_masses.py    # RDKit monoisotopic masses for train structures
uv run python scripts/build_pool_fps.py        # Morgan fingerprints for the full candidate pool
uv run python scripts/export_scorer_json.py    # learned scorer for the kernel
```
