"""Compute predictive value of imaging features for cognition scores.

Support vector regression is chosen as prediction algorithm. Hyperparameters C, epsilon, and gamma
are chosen for each model with Bayesian optimisation.

Requirements:
    - valid subjects were identified and latent variables were stored in a_collect_data.py

Outputs:
    - csv with R² scores in the hold-out test set per modality and iteration
"""

# %%
import sys
from pathlib import Path

# Add parent folder to path to import utils
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from clinical_validation_nihss.utils import (
    fit_svr_bayes_opt,
    load_vectorised_images,
    train_test_split_indices,
)
from joblib import Parallel, delayed
from sklearn.metrics import r2_score

N_PREDICTION_REPS = 250  # number of repeated cross-validations per data modality
# set minimum number of lesions per voxel to be considered informative and
# included in the regression in the full voxel-wise data condition
MIN_LESION_THRESHOLD = 10
TEST_SIZE_RATIO = 0.2
N_WORKERS = 10

SUBJECT_ID = "SubjectID"
NIHSS_24H = "NIHSS_24h"
NIFTI_PATH = "NiftiPath"
AGE = "Age"
SELECTIVE_ATTENTION = "Selective attention (total omissions)"
WORD_FLUENCY = "phon_alt_g_r_Roh"

FULL_DATA_CSV = Path(__file__).parent / "a_collect_data.csv"

LATENTS_ALL_METHODS = (
    Path(__file__).parent
    / "output_compressed_images_cognition_sample"
    / "compressed_images_cognition.npz"
)

# %%
# load clinical and latent imaging data
scores_df = pd.read_csv(FULL_DATA_CSV, sep=";")
# both variables contain missing values (between 1 and 4); for simplicity, these are included
# entirely
included_subjects_idx = ~np.isnan(scores_df[SELECTIVE_ATTENTION]) & ~np.isnan(
    scores_df[WORD_FLUENCY]
)

lesion_nifti_path_list = scores_df[NIFTI_PATH][included_subjects_idx].tolist()

latents_all_methods = np.load(
    LATENTS_ALL_METHODS
)  # gives lazy access handle to the arrays

voxelwise_images_cont = load_vectorised_images(lesion_path_list=lesion_nifti_path_list)
voxelwise_images_binary = (voxelwise_images_cont > 0).astype(int)

# target scores are selective attention and word fluency
# In line with a previous study, selective attention scores (but not word fluency scores) are log
# transformed to better handle the data distribution
target_selective_attention = np.log1p(
    scores_df[SELECTIVE_ATTENTION][included_subjects_idx].to_numpy()
)
target_word_fluency = scores_df[WORD_FLUENCY][included_subjects_idx].to_numpy()

# add baseline variable age
baseline_vars = np.stack([scores_df[AGE][included_subjects_idx].values], axis=1)


latent_modalities = {
    f"{key}": latents_all_methods[key] for key in latents_all_methods.files
}

# Only include valid subjects in all imaging modalities
for key in latent_modalities:
    latent_modalities[key] = latent_modalities[key][included_subjects_idx]

## voxelwise data were not used, but could be merged into the dict here
# voxelwise_modalities = {
#     "voxelwise_cont": voxelwise_images_cont,
#     "voxelwise_binary": voxelwise_images_binary,
# }
# imaging_modalities = {**latent_modalities, **voxelwise_modalities}
imaging_modalities = {**latent_modalities}

# Augment each modality with baseline vars age and lesion volume
imaging_modalities = {
    modality: np.concatenate([X, baseline_vars], axis=1)
    for modality, X in imaging_modalities.items()
}


# %%
# define functions for parallelisation
def _run_prediction_selective_attention(seed):
    results = []
    train_idx, test_idx = train_test_split_indices(
        len(target_selective_attention), TEST_SIZE_RATIO, seed
    )
    y_train, y_test = (
        target_selective_attention[train_idx],
        target_selective_attention[test_idx],
    )

    for modality, x in imaging_modalities.items():

        x_train, x_test = x[train_idx], x[test_idx]
        model = fit_svr_bayes_opt(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        results.append({"modality": modality, "split": seed, "r2": r2})
    return results


def _run_prediction_word_fluency(seed):
    results = []
    train_idx, test_idx = train_test_split_indices(
        len(target_word_fluency), TEST_SIZE_RATIO, seed
    )
    y_train, y_test = (
        target_word_fluency[train_idx],
        target_word_fluency[test_idx],
    )

    for modality, x in imaging_modalities.items():

        x_train, x_test = x[train_idx], x[test_idx]
        model = fit_svr_bayes_opt(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        results.append({"modality": modality, "split": seed, "r2": r2})
    return results


# %%
# perform parallelised analysis for selective attention

all_results_selective_attention = Parallel(n_jobs=N_WORKERS, verbose=10)(
    delayed(_run_prediction_selective_attention)(seed)
    for seed in range(N_PREDICTION_REPS)
)
results = [item for sublist in all_results_selective_attention for item in sublist]

base_name = Path(__file__).stem
filename_specifier = "selective_attention"
output_name = Path(__file__).with_name(f"{base_name}_{filename_specifier}.csv")
pd.DataFrame(results).to_csv(output_name, index=False)

# %%
# print summary for selective attention
results_df = pd.DataFrame(results)
for modality in results_df["modality"].unique():
    temp = results_df[results_df["modality"] == modality]
    mean_r2 = temp["r2"].mean()
    print(f"{modality} r2: {mean_r2}")

    # %%
# perform parallelised analysis for word fluency

all_results_word_fluency = Parallel(n_jobs=N_WORKERS, verbose=10)(
    delayed(_run_prediction_word_fluency)(seed) for seed in range(N_PREDICTION_REPS)
)
results = [item for sublist in all_results_word_fluency for item in sublist]

base_name = Path(__file__).stem
filename_specifier = "word_fluency"
output_name = Path(__file__).with_name(f"{base_name}_{filename_specifier}.csv")
pd.DataFrame(results).to_csv(output_name, index=False)

# %%
# print summary for word fluency
results_df = pd.DataFrame(results)
for modality in results_df["modality"].unique():
    temp = results_df[results_df["modality"] == modality]
    mean_r2 = temp["r2"].mean()
    print(f"{modality} r2: {mean_r2}")

# %%
