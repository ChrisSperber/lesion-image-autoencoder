"""Statistical analysis and visualisation of cognition validation results.

Requirements:
    - b_elastic_net_cognition.py and c_svr_cognition.py were run

Outputs:
    - json file with statistical results
    - results plot for publication
"""

# %%
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

ELASTIC_NET_SELATTENTION_CSV_PATH = (
    Path(__file__).parent / "b_elastic_net_cognition_selective_attention.csv"
)
ELASTIC_NET_WORDFLUENCY_CSV_PATH = (
    Path(__file__).parent / "b_elastic_net_cognition_word_fluency.csv"
)

SVR_SELATTENTION_CSV_PATH = (
    Path(__file__).parent / "c_svr_cognition_selective_attention.csv"
)
SVR_WORDFLUENCY_CSV_PATH = Path(__file__).parent / "c_svr_cognition_word_fluency.csv"

MODALITY_NAME_MAP = {
    "images_compressed_pca": "PCA",
    "images_compressed_svd": "Truncated SVD",
    "images_compressed_nmf": "NMF",
    "images_compressed_deep_ae": "Deep Autoencoder",
}
MODEL_NAME_MAP = {
    "elastic_net_selattention": "Selective Attention - Elastic Net Regression",
    "svr_selattention": "Selective Attention - SVR",
    "elastic_net_wordfluency": "Word Fluency - Elastic Net Regression",
    "svr_wordfluency": "Word Fluency - SVR",
}
METHODS_PLOT_NAMES = ["PCA", "SVD", "NMF", "DeepAE"]
ORDER = ["PCA", "Truncated SVD", "NMF", "Deep Autoencoder"]

METHODS = ["elastic_net", "svr"]
COGNITION_DEFICITS = ["selective_attention", "word_fluency"]

MODALITY = "modality"
R2_SCORE = "r2"
SPLIT = "split"

STAT_RESULTS_JSON = Path(__file__).parent / f"{Path(__file__).stem}_results.json"

METRICS_DESCRIPTIVE_ANALYSIS = [
    R2_SCORE,
]

# %%
results_df_selattention_elastic_net = pd.read_csv(
    ELASTIC_NET_SELATTENTION_CSV_PATH, sep=","
)
results_df_wordfluency_elastic_net = pd.read_csv(
    ELASTIC_NET_WORDFLUENCY_CSV_PATH, sep=","
)
results_df_selattention_svr = pd.read_csv(SVR_SELATTENTION_CSV_PATH, sep=",")
results_df_wordfluency_svr = pd.read_csv(SVR_WORDFLUENCY_CSV_PATH, sep=",")

# %%
# very few predictions with elastic net regression on data transformed by the NMF model
# failed with extremely large residuals and hence highly negative R2 values
# these values are handled by winsorizing extreme values


def _clip_column(df, col, q_low=0.01, q_high=0.99):
    low, high = df[col].quantile([q_low, q_high])
    df[col] = df[col].clip(lower=low, upper=high)


# R2 clip extreme low vals
_clip_column(results_df_selattention_elastic_net, R2_SCORE, 0.01, 1)
_clip_column(results_df_wordfluency_elastic_net, R2_SCORE, 0.01, 1)

# NOTE: Data were visually inspected and extreme outliers are indeed removed

results_dfs = {
    "elastic_net_selattention": results_df_selattention_elastic_net,
    "svr_selattention": results_df_selattention_svr,
    "elastic_net_wordfluency": results_df_wordfluency_elastic_net,
    "svr_wordfluency": results_df_wordfluency_svr,
}

# %%
# main analyses
statistical_results = {}
formatted_tables = defaultdict(dict)  # model_name -> metric table (DataFrame)

for model_name, df in results_dfs.items():
    print(f"Analysing {model_name}")

    ######
    # Plot
    df["mod_pretty"] = pd.Categorical(
        df[MODALITY].map(MODALITY_NAME_MAP), categories=ORDER, ordered=True
    )

    model_name_pretty = MODEL_NAME_MAP[model_name]

    plt.figure(figsize=(5, 6))
    ax = sns.swarmplot(data=df, x="mod_pretty", y=R2_SCORE, size=2, order=ORDER)

    means = df.groupby("mod_pretty")[R2_SCORE].mean().reindex(ORDER)
    for xtick, modality in enumerate(means.index):
        mean_val = means[modality]
        ax.plot([xtick - 0.2, xtick + 0.2], [mean_val, mean_val], color="black", lw=2)

    title_str = model_name_pretty
    ax.set_xlabel("")
    ax.set_xticklabels(METHODS_PLOT_NAMES)
    plt.title(title_str)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{Path(__file__).stem}_{model_name}_r2_swarmplot.png", dpi=300)

    # Show the plot
    plt.show()
    ######

    ######
    # Statistical Analysis
    print(f"Running ANOVA and posthoc tests for {model_name}")
    model_results = {}

    # Pivot table to wide format: rows = subjects, columns = modalities
    pivoted = df.pivot(index=SPLIT, columns="mod_pretty", values=R2_SCORE)

    # Prepare long format
    long_df = pivoted.reset_index().melt(
        id_vars=SPLIT, var_name="mod_pretty", value_name=R2_SCORE
    )

    # Repeated Measures ANOVA
    aovrm = AnovaRM(data=long_df, depvar=R2_SCORE, subject=SPLIT, within=["mod_pretty"])
    anova_res = aovrm.fit()

    anova_table = anova_res.anova_table
    anova_stat = anova_table["F Value"].iloc[0]
    anova_p = anova_table["Pr > F"].iloc[0]

    # Pairwise t-tests (paired, repeated measures)
    raw_pvals = []
    comparisons = []

    for m1, m2 in combinations(pivoted.columns, 2):
        stat, p = ttest_rel(pivoted[m1], pivoted[m2])
        raw_pvals.append(p)
        comparisons.append((m1, m2, stat, p))

    # Multiple testing correction (Bonferroni-Holm)
    corrected = multipletests(raw_pvals, method="holm")
    corrected_pvals = corrected[1]

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

    model_results = {
        "anova": {
            "statistic": anova_stat,
            "p_value": anova_p,
        },
        "t_test_posthoc": posthoc_results,
    }

    statistical_results[model_name] = model_results

    ######
    # Descriptive Statistics
    group_stats = {}
    for mod_name, group in df.groupby("mod_pretty"):
        metric_stats = {}
        for metric in METRICS_DESCRIPTIVE_ANALYSIS:
            metric_stats[metric] = {
                "mean": group[metric].mean(),
                "std": group[metric].std(),
            }
        group_stats[mod_name] = metric_stats

    statistical_results[model_name]["descriptive_stats"] = group_stats

    ######
    # Print descriptives table to interactive window for copypasting
    table_data = defaultdict(dict)  # metric -> column name -> value

    for mod_name, group in df.groupby("mod_pretty"):
        for metric in METRICS_DESCRIPTIVE_ANALYSIS:
            mean = group[metric].mean()
            std = group[metric].std()
            table_data[metric][mod_name] = f"{mean:.2f} ± {std:.2f}"

    # Convert to DataFrame and store
    formatted_df = pd.DataFrame.from_dict(table_data, orient="index")
    formatted_tables[model_name] = formatted_df
    print(f"\nDescriptive statistics for {model_name}:\n")
    display(formatted_df)  # Works in Jupyter and VSCode Interactive Window

# Export to JSON
with open(STAT_RESULTS_JSON, "w") as f:
    json.dump(statistical_results, f, indent=2)

# %%
