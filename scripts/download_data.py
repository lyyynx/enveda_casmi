"""Download the Enveda CASMI26 competition data via kagglehub into the project data folder."""

from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Download latest version directly into the project data folder
path = kagglehub.competition_download(
    "enveda-CASMI26-molecule-id-mass-spectra",
    path_override=str(DATA_DIR),
)

print("Path to competition files:", path)
