"""Train the candidate-level contrastive likelihood scorer.

Aggregates per-match rows to one row per (spectrum, candidate), z-scores
features within each spectrum's candidate set, fits a logistic model, and
reports validation AUC and within-spectrum top-1 accuracy.

Usage:
    uv run python scripts/train_scorer.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from enveda_casmi26.pipeline import ce_bucket
from enveda_casmi26.scoring import NUMERIC_FEATURES, CandidateScorer, candidate_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "scorer.joblib"


def build_candidate_table(dataset: pd.DataFrame, spectra_info: pd.DataFrame,
                          pred_fps: dict, cand_fps: dict) -> pd.DataFrame:
    rows = []
    for (spectrum_id, candidate_key), group in dataset.groupby(["spectrum_id", "candidate_key"], sort=False):
        info = spectra_info.loc[spectrum_id]
        row = candidate_features(
            group,
            n_peaks=info["n_peaks"],
            precursor_mz=info["precursor_mz"],
            adduct=info["adduct"],
            ionization_mode=info["ionization_mode"],
            total_intensity=info["total_intensity"],
            ce_bucket=info["ce_bucket"],
            fp_tanimoto=fp_tanimoto(pred_fps.get(spectrum_id), cand_fps.get(candidate_key)),
        )
        row["spectrum_id"] = spectrum_id
        row["is_decoy"] = int(group["is_decoy"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def fp_tanimoto(a, b) -> float:
    if a is None or b is None:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    union = int(a.sum() + b.sum() - inter)
    return inter / union if union > 0 else 0.0


def compute_pred_fps(spectrum_ids) -> dict:
    """Predict a 2048-bit fingerprint per query spectrum (encoder + fp head)."""
    import torch

    from enveda_casmi26.fp_model import encode_batch, load_fp_model, prep_spectrum

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, head = load_fp_model(PROJECT_ROOT / "models" / "denovo_v1.pt",
                                PROJECT_ROOT / "models" / "fp_head.pt", device)

    from enveda_casmi26.fragmentation import ADDUCTS, ADDUCT_ALIASES

    tc = pd.read_parquet(DATA_DIR / "train_clean.parquet",
                         columns=["inchikey14", "precursor_mz", "adduct",
                                  "ms2_mzs", "ms2_normalized_intensities",
                                  "num_peaks_clean", "drop_reason"])
    tc = tc[(tc["drop_reason"] == "") & (tc["num_peaks_clean"] >= 10)
            & (tc["adduct"].isin(set(ADDUCTS) | set(ADDUCT_ALIASES)))
            & tc["inchikey14"].notna()]
    tc = tc.drop_duplicates(subset="inchikey14", keep="first")
    tc = tc[tc["inchikey14"].isin(spectrum_ids)]
    print(f"encoding {len(tc)} query spectra for predicted fingerprints ...")

    out: dict = {}
    mz_list, int_list, prec_list, ids = [], [], [], []
    for r in tc.itertuples(index=False):
        mzs, ints = prep_spectrum(r.ms2_mzs, r.ms2_normalized_intensities)
        if len(mzs) == 0:
            continue
        mz_list.append(mzs)
        int_list.append(ints)
        prec_list.append(float(r.precursor_mz))
        ids.append(r.inchikey14)
    if mz_list:
        pooled = encode_batch(model, mz_list, int_list, prec_list, device)
        with torch.no_grad():
            probs = torch.sigmoid(head(torch.from_numpy(pooled).to(device))).cpu().numpy()
        for k, sid in enumerate(ids):
            out[sid] = probs[k] > 0.5
    return out


def compute_cand_fps(candidate_keys) -> dict:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    tc = pd.read_parquet(DATA_DIR / "train_clean.parquet",
                         columns=["inchikey14", "normalized_smiles"])
    tc = tc.dropna(subset=["normalized_smiles"]).drop_duplicates(subset="inchikey14")
    tc = tc[tc["inchikey14"].isin(candidate_keys)]
    out: dict = {}
    for r in tc.itertuples(index=False):
        mol = Chem.MolFromSmiles(r.normalized_smiles)
        if mol is None:
            continue
        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros(2048, dtype=bool)
        for b in fp.GetOnBits():
            arr[b] = True
        out[r.inchikey14] = arr
    print(f"fingerprinted {len(out)} candidate structures")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset", default="match_dataset_default.parquet")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--drop-ce", action="store_true",
                        help="ablation: exclude CE deviation features from the model")
    args = parser.parse_args()

    if args.drop_ce:
        import enveda_casmi26.scoring as scoring_module

        scoring_module.NUMERIC_FEATURES = [
            f for f in scoring_module.NUMERIC_FEATURES if not f.startswith("ce_")]
        globals()["NUMERIC_FEATURES"] = scoring_module.NUMERIC_FEATURES
        print("ablation: CE features excluded")

    print("Loading match dataset ...")
    dataset = pd.read_parquet(DATA_DIR / args.dataset)
    print("Loading spectrum info from cleaned training data ...")
    tc = pd.read_parquet(
        DATA_DIR / "train_clean.parquet",
        columns=["inchikey14", "precursor_mz", "adduct", "ionization_mode", "num_peaks_clean",
                 "ms2_normalized_intensities"],
    )
    ce = pq.read_table(DATA_DIR / "train.parquet",
                       columns=["collision_energy_orig",
                                "collision_energy_orig_units"]).to_pandas()
    assert len(ce) == len(tc), "train CE columns do not align with train_clean"
    tc["ce_bucket"] = [ce_bucket(v, u) for v, u in
                       zip(ce["collision_energy_orig"],
                           ce["collision_energy_orig_units"])]
    tc = tc.set_index("inchikey14")
    spectra_info = pd.DataFrame({
        "n_peaks": tc["num_peaks_clean"],
        "precursor_mz": tc["precursor_mz"],
        "adduct": tc["adduct"],
        "ionization_mode": tc["ionization_mode"],
        "ce_bucket": tc["ce_bucket"],
        "total_intensity": [float(np.sum(i)) for i in tc["ms2_normalized_intensities"]],
    })
    spectra_info = spectra_info[~spectra_info.index.duplicated(keep="first")]

    candidates = build_candidate_table(dataset, spectra_info,
                                       pred_fps=compute_pred_fps(dataset["spectrum_id"].unique()),
                                       cand_fps=compute_cand_fps(dataset["candidate_key"].unique()))
    print(f"candidate rows: {len(candidates):,} "
          f"({(candidates['is_decoy'] == 0).sum():,} true / {(candidates['is_decoy'] == 1).sum():,} decoy) "
          f"over {candidates['spectrum_id'].nunique()} spectra")

    spectra = np.asarray(candidates["spectrum_id"].unique(), dtype=object)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(spectra)
    n_valid = int(len(spectra) * args.valid_frac)
    valid_ids = set(spectra[:n_valid])

    train = candidates[~candidates["spectrum_id"].isin(valid_ids)]
    valid = candidates[candidates["spectrum_id"].isin(valid_ids)]
    print(f"train: {len(train):,} candidates from {train['spectrum_id'].nunique()} spectra")
    print(f"valid: {len(valid):,} candidates from {valid['spectrum_id'].nunique()} spectra")

    def contrasted(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        xs, ys, ids = [], [], []
        for spectrum_id, group in frame.groupby("spectrum_id", sort=False):
            xs.append(CandidateScorer.contrast_within_set(group))
            ys.append(group["is_decoy"].to_numpy())
            ids.append(np.full(len(group), spectrum_id))
        return pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(ids)

    train_x, train_y, _ = contrasted(train)
    valid_x, valid_y, valid_ids_arr = contrasted(valid)

    scorer = CandidateScorer()
    scorer.fit(train_x, train_y)
    probs = scorer.score_set(valid_x.to_dict("records"))
    print(f"\nvalidation ROC-AUC (per candidate, contrasted): {roc_auc_score(valid_y, 1 - probs):.3f}")

    valid_x = valid_x.assign(prob=probs, spectrum_id=valid_ids_arr, is_decoy=valid_y)
    top1 = top3 = n_eval = 0
    for _, group in valid_x.groupby("spectrum_id"):
        true_probs = group.loc[group["is_decoy"] == 0, "prob"]
        decoy_probs = group.loc[group["is_decoy"] == 1, "prob"]
        if true_probs.empty or decoy_probs.empty:
            continue
        n_eval += 1
        rank = 1 + int((decoy_probs >= true_probs.iloc[0]).sum())
        top1 += rank == 1
        top3 += rank <= 3
    print(f"validation top-1: {top1 / n_eval:.1%}   top-3: {top3 / n_eval:.1%}   ({n_eval} spectra)")

    cov1 = 0
    cov_eval = 0
    for _, group in valid.groupby("spectrum_id"):
        true_cov = group.loc[group["is_decoy"] == 0, "coverage"]
        decoy_cov = group.loc[group["is_decoy"] == 1, "coverage"]
        if true_cov.empty or decoy_cov.empty:
            continue
        cov_eval += 1
        cov1 += int((decoy_cov < true_cov.iloc[0]).all())
    print(f"sanity - top-1 by coverage alone: {cov1 / cov_eval:.1%} ({cov_eval} spectra)")

    model_path = Path(args.model_path) if args.model_path else MODEL_PATH
    scorer.save(model_path)
    print(f"model saved to {model_path}")

    coef = pd.Series(scorer.model.coef_[0], index=scorer.columns)
    print("\nlearned within-spectrum effects (log-odds):")
    print(coef[NUMERIC_FEATURES].sort_values(key=lambda s: s.abs(), ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
