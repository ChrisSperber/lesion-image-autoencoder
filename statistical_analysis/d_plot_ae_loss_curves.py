"""Create plot of train/validation loss of autoencoders."""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAINED_MODELS_DIR = (
    Path(__file__).parents[1]
    / "dimensionality_reduction"
    / "autoencoder_utils"
    / "outputs"
)

FINAL_MODEL_BINARY_DIR = "output_deep_nonlinear_binary_input_20250512_0926"
FINAL_MODEL_CONT_DIR = "output_deep_nonlinear_continuous_input_20250512_1139"

METRICS_FILE = "metrics.json"

# %%
# fetch data about train/validation loss
binary_metrics_json = TRAINED_MODELS_DIR / FINAL_MODEL_BINARY_DIR / METRICS_FILE
cont_metrics_json = TRAINED_MODELS_DIR / FINAL_MODEL_CONT_DIR / METRICS_FILE

binary_metrics = pd.read_json(binary_metrics_json)
cont_metrics = pd.read_json(cont_metrics_json)

# %%
# define plotting function


def plot_training_curves(
    df: pd.DataFrame, patience: int = 15, title: str = None, savepath: Path = None
):
    """Plot the training/validation curve.

    Args:
        df: Dataframe with columns "val_loss" and "train_loss"
        patience: Patience criterion for early stopping. Defaults to 15.
        title: Figure title.
        savepath: If provided, figure is saved under this path.

    """
    # Ensure epochs start at 1 for nicer ticks
    epochs = np.arange(1, len(df) + 1)

    # Find best epoch by validation loss
    best_idx = int(df["val_loss"].idxmin())
    # If df index starts at 0, convert to epoch number
    if df.index.min() == 0:
        best_epoch = best_idx + 1
    else:
        best_epoch = int(best_idx)

    stop_epoch = min(best_epoch + patience, len(df))

    plt.figure()
    plt.plot(epochs, df["train_loss"].values, label="Train loss")
    plt.plot(epochs, df["val_loss"].values, label="Validation loss")

    # Mark best epoch
    plt.axvline(
        best_epoch, linestyle="--", linewidth=1, label=f"Best val (epoch {best_epoch})"
    )
    plt.scatter([best_epoch], [df.loc[df.index[best_epoch - 1], "val_loss"]], zorder=3)

    # Shade the no-improvement window (patience)
    if stop_epoch > best_epoch:
        plt.axvspan(
            best_epoch,
            stop_epoch,
            alpha=0.15,
            label=f"No improvement (patience={patience})",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# %%
# plot binary training data
savepath = Path(__file__).with_name(Path(__file__).stem + "_binary" + ".png")
plot_training_curves(
    binary_metrics,
    patience=15,
    title="Autoencoder training binary data",
    savepath=savepath,
)

# %%
# plot continuous training data
savepath = Path(__file__).with_name(Path(__file__).stem + "_continuous" + ".png")
plot_training_curves(
    cont_metrics,
    patience=15,
    title="Autoencoder training binary data",
    savepath=savepath,
)

# %%
