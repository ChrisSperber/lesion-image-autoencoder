"""Compute predictive value of imaging features for NIHSS at ~24h post-stroke.

Elastic net regression is chosen as prediction algorithm, a linear regression method with
regularisation. Hyperparameters L1 ratio and alpha define the regularisation and are chosen for each
model with Bayesian optimisation.

Requirements:
    - valid subjects were identified in a_collect_data_and_demographics.py
    - latent variables were stored for baseline methods pca/truncated svd/nmf and deep AEs in
        Code/dimensionality_reduction

Outputs:
    - csv with R² scores in the hold-out test set per modality and iteration
"""

# %%
import time  # for debugging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from utils import (
    fit_elastic_net_bayes_opt,
    load_masked_vectorised_images,
    train_test_split_indices,
)

N_PREDICTION_REPS = 100  # number of repeated cross-validations per data modality
# set minimum number of lesions per voxel to be considered informative and
# included in the regression in the full voxel-wise data condition
MIN_LESION_THRESHOLD = 10
TEST_SIZE_RATIO = 0.2

SUBJECT_ID = "SubjectID"
NIHSS_24H = "NIHSS_24h"
NIFTI_PATH = "NiftiPath"
LESION_VOLUME_ML = "LesionSizeML_p02"
AGE = "Age"

NIHSS_SCORES_CSV = Path(__file__).parent / "a_collect_data_and_demographics.csv"

FULL_DATA_CSV = (
    Path(__file__).parents[1]
    / "data_collection"
    / "a_verify_and_collect_lesion_data.csv"
)

LATENTS_BASELINE_METHODS = (
    Path(__file__).parents[1]
    / "dimensionality_reduction"
    / "output_compressed_images"
    / "compressed_images_pca_svd_nmf.npz"
)
LATENTS_DEEP_AE = (
    Path(__file__).parents[1]
    / "dimensionality_reduction"
    / "output_compressed_images"
    / "compressed_images_deep_ae.npz"
)

# %%
# load clinical and latent imaging data
nihss_df = pd.read_csv(NIHSS_SCORES_CSV, sep=";")
lesion_nifti_path_list = nihss_df[NIFTI_PATH].tolist()

latents_baseline = np.load(
    LATENTS_BASELINE_METHODS
)  # gives lazy access handle to the arrays
latents_deep_ae = np.load(LATENTS_DEEP_AE)

voxelwise_images_cont = load_masked_vectorised_images(
    lesion_path_list=lesion_nifti_path_list, min_lesion_threshold=MIN_LESION_THRESHOLD
)
voxelwise_images_binary = (voxelwise_images_cont > 0).astype(int)

# in line with previous studies, nihss scores are log transformed to better handle the data
# distribution
target_nihss = np.log1p(nihss_df[NIHSS_24H].to_numpy())

baseline_vars = np.stack(
    [nihss_df[AGE].values, nihss_df[LESION_VOLUME_ML].values], axis=1
)

# %%
# collect all imaging data in a dict
# Load all latent arrays from the npz files
# WARNING: The latent variables contain data for all 1080 subject, but only the 935 included
# datasets must be used in the prediction

# full dataset demographics are loaded to derive an index of included subjects
full_data_df = pd.read_csv(FULL_DATA_CSV, sep=";")
# merge nihss scores into full data df
full_data_df = full_data_df.merge(
    nihss_df[[SUBJECT_ID, NIHSS_24H]], on=SUBJECT_ID, how="left"
)
included_subjects_idx = full_data_df[NIHSS_24H].notna().to_numpy()

latent_modalities = {
    f"baseline_{key}": latents_baseline[key] for key in latents_baseline.files
}
latent_modalities.update(
    {f"deep_ae_{key}": latents_deep_ae[key] for key in latents_deep_ae.files}
)
# Only include valid subjects in all imaging modalities
for key in latent_modalities:
    latent_modalities[key] = latent_modalities[key][included_subjects_idx]

# Add voxelwise modalities
voxelwise_modalities = {
    "voxelwise_cont": voxelwise_images_cont,
    "voxelwise_binary": voxelwise_images_binary,
}

imaging_modalities = {**latent_modalities, **voxelwise_modalities}

# Augment each modality with baseline vars age and lesion volume
imaging_modalities = {
    modality: np.concatenate([X, baseline_vars], axis=1)
    for modality, X in imaging_modalities.items()
}

# %%
results = []

for seed in range(0, N_PREDICTION_REPS):
    train_idx, test_idx = train_test_split_indices(
        len(target_nihss), TEST_SIZE_RATIO, seed
    )
    y_train, y_test = target_nihss[train_idx], target_nihss[test_idx]

    for modality, X in imaging_modalities.items():
        print(f"Fitting {modality} on split {seed}")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")  # for debugging
        print(current_time)  # for debugging
        X_train, X_test = X[train_idx], X[test_idx]

        model = fit_elastic_net_bayes_opt(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        results.append({"modality": modality, "split": seed, "r2": r2})

# %%
output_name = Path(__file__).with_suffix(".csv")
pd.DataFrame(results).to_csv(output_name, index=False)

# %%
# print summary
results_df = pd.DataFrame(results)
for modality in results_df["modality"].unique():
    temp = results_df[results_df["modality"] == modality]
    mean_r2 = temp["r2"].mean()
    print(f"{modality} r2: {mean_r2}")

# %%
