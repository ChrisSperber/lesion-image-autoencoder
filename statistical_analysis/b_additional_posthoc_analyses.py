"""Additional posthoc analysis on relation of lesion size and reconstruction errors.

These analyses were not planned and were conducted after conducting the main analyses to potentially
better understand the main results.

Requirements:
    - simple_feature_reduction.py was succesfully run and reconstruction errors were stored in a csv
    - all autoencoders were succesfully trained and their reconstruction errors were evaluated with
        evaluate_autoencoders.py and stored in a csv

Outputs:
    - json file documenting statistical results

"""

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

DIMENSIONALITY_REDUCTION_DIR = Path(__file__).parents[1] / "dimensionality_reduction"
RECONSTRUCTION_ERRORS_PCA_SVD_NMF = (
    DIMENSIONALITY_REDUCTION_DIR / "simple_feature_reduction.csv"
)
RECONSTRUCTION_ERRORS_AUTOENCODERS = (
    DIMENSIONALITY_REDUCTION_DIR / "evaluate_autoencoders.csv"
)

SUBJECT_ID = "SubjectID"
LESION_SIZE = "LesionSizeML_p02"

DATA_COLLECTION_DIR = Path(__file__).parents[1] / "data_collection"
SAMPLE_DATA_CSV = "a_verify_and_collect_lesion_data.csv"

methods = ["PCA", "TruncSVD", "NMF", "DeepAE"]
data_types = ["Binary", "Continuous"]

RNG_SEED = 9001

# %%
# read csv files with reconstruction errors, unify column names
recon_errors_df_baseline = pd.read_csv(RECONSTRUCTION_ERRORS_PCA_SVD_NMF, delimiter=";")
recon_errors_df_baseline.columns = recon_errors_df_baseline.columns.str.replace(
    "Reconstruction", "", regex=False
)

recon_errors_df_autoencoders = pd.read_csv(
    RECONSTRUCTION_ERRORS_AUTOENCODERS, delimiter=";"
)
recon_errors_df = pd.merge(
    recon_errors_df_baseline,
    recon_errors_df_autoencoders,
    on=SUBJECT_ID,
    validate="1:1",
)

# %%
# read lesion size from sample data csv
sample_data_df = pd.read_csv(DATA_COLLECTION_DIR / SAMPLE_DATA_CSV, delimiter=";")

recon_errors_df = recon_errors_df.merge(
    sample_data_df[[SUBJECT_ID, LESION_SIZE]],
    on=SUBJECT_ID,
    how="left",
    validate="one_to_one",
)


# %%
# define helper function to bootstrap CIs
def bootstrap_kendall_ci(
    x, y, n_bootstrap: int = 1000, ci: int = 95, random_state: int = RNG_SEED
) -> tuple[float, float]:
    """Bootstrap confidence interval of Kendall's Tau.

    Args:
        x: First variable (1D array-like).
        y: Second variable (1D array-like).
        n_bootstrap (optional): Number of bootstraps. Defaults to 1000.
        ci (optional): % confidence interval. Defaults to 95%.
        random_state (optional): random seed.

    Returns:
        Lower and upper CI bounds.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    taus = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        tau, _ = kendalltau(x[idx], y[idx])
        taus.append(tau)
    lower_ci = np.percentile(taus, (100 - ci) / 2)
    upper_ci = np.percentile(taus, 100 - (100 - ci) / 2)
    return lower_ci, upper_ci


# %%

statistical_results = []

for data_type in data_types:
    for method in methods:
        colname = f"{data_type}{method}"

        x = recon_errors_df[colname]
        y = recon_errors_df[LESION_SIZE]
        tau, p = kendalltau(x, y)
        lower_ci, upper_ci = bootstrap_kendall_ci(x, y)

        results = {
            "Data_type": data_type,
            "Latent_method": method,
            "Kendall's Tau": tau,
            "pval": p,
            "95%-CI": [lower_ci, upper_ci],
        }

        statistical_results.append(results)

# export to JSON
output_path = Path(__file__).with_suffix(".json")

with open(output_path, "w") as f:
    json.dump(statistical_results, f, indent=2)

# %%
