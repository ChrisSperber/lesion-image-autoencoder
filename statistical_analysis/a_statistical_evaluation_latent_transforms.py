"""Statistical evaluation of autoencoder and comparison methods performance.

Requirements:
    - simple_feature_reduction.py was succesfully run and reconstruction errors were stored in a csv
    - all autoencoders were succesfully trained and their reconstruction errors were evaluated with
        evaluate_autoencoders.py and stored in a csv

Outputs:
    - file statistical_results.json documenting statistical results
    - plots of reconstruction errors

"""

# %%
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

DIMENSIONALITY_REDUCTION_DIR = Path(__file__).parents[1] / "dimensionality_reduction"
RECONSTRUCTION_ERRORS_PCA_SVD_NMF = (
    DIMENSIONALITY_REDUCTION_DIR / "simple_feature_reduction.csv"
)
RECONSTRUCTION_ERRORS_AUTOENCODERS = (
    DIMENSIONALITY_REDUCTION_DIR / "evaluate_autoencoders.csv"
)
STAT_RESULTS_JSON = Path(__file__).parent / "statistical_results.json"

SUBJECT_ID = "SubjectID"

ALPHA = 0.05  # statistical significance threshold for visualisation

methods = ["PCA", "TruncSVD", "NMF", "DeepAE"]
data_types = ["Binary", "Continuous"]

METHODS_PLOT_NAMES = ["PCA", "SVD", "NMF", "DeepAE"]

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
statistical_results = {}

for data_type in data_types:
    relevant_columns = [f"{data_type}{method}" for method in methods]
    data = recon_errors_df[relevant_columns]

    # Friedman test
    friedman_stat, friedman_p = friedmanchisquare(
        *[data[col] for col in relevant_columns]
    )
    print(f"{data_type} data - Friedman test p-value: {friedman_p:.4g}")

    # post-hoc comparisons via Wilcoxon tests
    raw_pvals = []
    comparisons = []

    for m1, m2 in combinations(relevant_columns, 2):
        stat, p = wilcoxon(data[m1], data[m2])
        raw_pvals.append(p)
        comparisons.append((m1, m2, stat, p))

    # multiple testing correction via Bonferroni-Holm
    corrected = multipletests(raw_pvals, method="holm")
    corrected_pvals = corrected[1]

    # format results
    posthoc_results = []
    for (m1, m2, stat, raw_p), corrected_p in zip(
        comparisons, corrected_pvals, strict=True
    ):
        posthoc_results.append(
            {
                "comparison": [m1, m2],
                "statistic": stat,
                "p_value_uncorrected": raw_p,
                "p_value_corrected": corrected_p,
            }
        )

    statistical_results[data_type] = {
        "friedman_test": {
            "statistic": friedman_stat,
            "p_value": friedman_p,
        },
        "wilcoxon_posthoc": posthoc_results,
    }

# export to JSON
with open(STAT_RESULTS_JSON, "w") as f:
    json.dump(statistical_results, f, indent=2)

# %%
# Visualisation
# Binary data
# restructure results into seaborn-friendly format
melted_df_binary = recon_errors_df.melt(
    id_vars=SUBJECT_ID,
    value_vars=[f"Binary{m}" for m in methods],
    var_name="Method",
    value_name="Reconstruction Dice Score",
)

plt.figure(figsize=(10, 6))
ax = sns.swarmplot(
    data=melted_df_binary, x="Method", y="Reconstruction Dice Score", size=2
)

# Add median lines manually
medians = melted_df_binary.groupby("Method")["Reconstruction Dice Score"].median()
for xtick, method in enumerate(methods):
    median = medians[f"Binary{method}"]
    ax.plot([xtick - 0.2, xtick + 0.2], [median, median], color="black", lw=2)

ax.set_xticklabels(METHODS_PLOT_NAMES)

plt.title("Binary Data Reconstruction Fidelity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results_binary_reconstruction.png", dpi=300, bbox_inches="tight")
plt.show()

# Continuous data
# restructure results into seaborn-friendly format
melted_df_continuous = recon_errors_df.melt(
    id_vars=SUBJECT_ID,
    value_vars=[f"Continuous{m}" for m in methods],
    var_name="Method",
    value_name="Reconstruction Error",
)

plt.figure(figsize=(10, 6))
ax = sns.swarmplot(
    data=melted_df_continuous, x="Method", y="Reconstruction Error", size=2
)

# Add median lines manually
medians = melted_df_continuous.groupby("Method")["Reconstruction Error"].median()
for xtick, method in enumerate(methods):
    median = medians[f"Continuous{method}"]
    ax.plot([xtick - 0.2, xtick + 0.2], [median, median], color="black", lw=2)

plt.ylim(bottom=0.0, top=0.035)
ax.set_xticklabels(METHODS_PLOT_NAMES)
ax.set_ylabel("Reconstruction Mean Absolute Error")

plt.title("Continuous Data Reconstruction Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results_continuous_reconstruction.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
