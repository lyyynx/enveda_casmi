# enveda-casmi26

Solution code for the Enveda CASMI26 molecule identification from mass spectra competition.

## Setup

```bash
uv sync
```

## Downloading the data

The competition data is not included in this repository (it is too large).
Download it with the helper script, which uses [kagglehub](https://pypi.org/project/kagglehub/)
to fetch the Kaggle competition dataset `enveda-CASMI26-molecule-id-mass-spectra`
into the `data/` folder:

```bash
python scripts/download_data.py
```

You will need Kaggle API credentials (`~/.kaggle/kaggle.json`) for the download to work.
Other derived datasets (`coconut_pool.parquet`, FP caches, etc.) are produced by the
scripts in `scripts/` (e.g. `build_coconut_pool.py`).
