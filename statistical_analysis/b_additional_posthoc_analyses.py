"""Additional posthoc analysis on relation of lesion size and reconstruction errors.

These analyses were not planned and were conducted after conducting the main analyses to potentially
better understand the main results.

Requirements:
    - simple_feature_reduction.py was succesfully run and reconstruction errors were stored in a csv
    - autoencoders were evaluated with evaluate_autoencoders.py and stored in a csv

Outputs:
    - json file documenting statistical results

"""

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

PLOT_TITLE_MAP = {
    "DeepAE": "Deep Autoencoder",
    "PCA": "Principal Component Analysis",
    "TruncSVD": "Singular Value Decomposition",
    "NMF": "Non-negative Matrix Factorisation",
}

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
# Visualize correlation lesion size (log-x) vs reconstruction fidelity (Dice) for Binary data
viz_df = recon_errors_df[
    [SUBJECT_ID, LESION_SIZE] + [f"Binary{m}" for m in methods]
].copy()
viz_df[LESION_SIZE] = viz_df[LESION_SIZE] / 1000  # convert lesion size to ml

# Pre-compute a dictionary: (method) -> (tau, p, ci_lo, ci_hi) for Binary
tau_lookup: dict[str, tuple[float, float, float, float]] = {}
for r in statistical_results:
    if r["Data_type"] == "Binary":
        ci_lo, ci_hi = r["95%-CI"]
        tau_lookup[r["Latent_method"]] = (
            r["Kendall's Tau"],
            r["pval"],
            ci_lo,
            ci_hi,
        )


def _get_log_ticks(xmin: float, xmax: float):
    candidates = np.array([0.5, 1, 10, 25, 100, 500], dtype=float)
    lo = candidates[candidates >= max(0.05, xmin)]
    ticks = lo[lo <= max(xmax, 0.05)]
    return ticks.tolist()


def _plot_panel(ax, x_raw, y_raw, method_name: str):
    # Keep only finite, positive x; finite y
    mask = np.isfinite(x_raw) & np.isfinite(y_raw) & (x_raw > 0)
    x = np.asarray(x_raw[mask])
    y = np.asarray(y_raw[mask])

    ax.scatter(x, y, s=14, alpha=0.5, linewidths=0, rasterized=True)
    ax.set_xscale("log")

    # Binned median trend (robust to outliers & ties)
    if x.size > 3:  # noqa: PLR2004
        n_bins = 12
        # guard if min==max
        xmin, xmax = x.min(), x.max()
        if xmax > xmin:
            bins = np.logspace(np.log10(xmin), np.log10(xmax), n_bins + 1)
            bin_ids = np.digitize(x, bins) - 1
            centers, medians = [], []
            for b in range(n_bins):
                sel = bin_ids == b
                if sel.any():
                    centers.append(np.sqrt(bins[b] * bins[b + 1]))  # geometric center
                    medians.append(np.nanmedian(y[sel]))
            if centers:
                ax.plot(centers, medians, marker="o", ms=4)

    # Limits
    ax.set_ylim(0, 1)
    ax.set_xlim(left=max(0.05, x.min() * 0.9), right=x.max() * 1.1)

    # Labels
    ax.set_xlabel("Lesion size (ml, log scale)")
    ax.set_ylabel("Reconstruction fidelity (Dice)")

    # Ticks & grid
    ticks = _get_log_ticks(x.min(), x.max())
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.2)

    # Annotation: Kendall tau, 95% CI, and p
    tau, p, ci_lo, ci_hi = tau_lookup.get(method_name, (np.nan, np.nan, np.nan, np.nan))
    ax.text(
        0.97,
        0.05,
        f"τ = {tau:.2f}\n95% CI [{ci_lo:.2f}, {ci_hi:.2f}]\np = {p:.1e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=4),
    )


# Build figure
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
axes = axes.ravel()

for ax, method in zip(axes, methods, strict=False):
    y = viz_df[f"Binary{method}"].astype(float)
    x = viz_df[LESION_SIZE].astype(float)
    _plot_panel(ax, x, y, method)
    ax.set_title(PLOT_TITLE_MAP[method])

fig.suptitle("Lesion size vs Reconstruction fidelity (Dice) — Binary data", fontsize=14)

out_png = Path(__file__).with_name("correl_lesion_size_vs_dice_binary.png")
fig.savefig(out_png, dpi=300)
plt.close(fig)

print(f"Saved figure to:\n - {out_png}")

# %%
