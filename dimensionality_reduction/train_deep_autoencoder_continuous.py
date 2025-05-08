"""Train deep nonlinear autoencoder for continuous images.

Continuous data are evaluated via L1Loss (i.e. MAE).
"""

# %%
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from autoencoder_utils.autoencoder_configs import (
    AUTOENCODER_OUTPUTS_DIR,
    LESION_WEIGHT_MULTIPLIER,
    N_LATENT_VARIABLES,
    TARGET_SHAPE_4CHANNEL,
    autoencoder_config,
)
from autoencoder_utils.autoencoder_dataset import LesionDataset
from autoencoder_utils.autoencoder_utils import AutoencoderType, get_batch_size_for_type
from autoencoder_utils.models.autoencoder_deep_nonlinear import Conv3dAutoencoder
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

AUTOENCODER_TYPE = AutoencoderType.DEEP_NONLINEAR_CONTINUOUS_INPUT

DATA_COLLECTION_DIR = "data_collection"
DATA_CSV = "a_verify_and_collect_lesion_data.csv"
LESION_PATH_COLUMN = "NiftiPath"

# %%
# load paths
data_df = pd.read_csv(
    Path(__file__).parents[1] / DATA_COLLECTION_DIR / DATA_CSV, sep=";"
)
nifti_path_list = data_df[LESION_PATH_COLUMN].tolist()

# %%
# assign training variables and create directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
AUTOENCODER_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
run_output_dir = (
    AUTOENCODER_OUTPUTS_DIR / f"output_{AUTOENCODER_TYPE.value}_{timestamp}"
)
run_output_dir.mkdir(parents=True, exist_ok=True)
epochs = autoencoder_config.epochs

batch_size = get_batch_size_for_type(
    autoencoder_type=AUTOENCODER_TYPE,
    batch_size_linear=autoencoder_config.batch_size_linear,
    batch_size_deep=autoencoder_config.batch_size_deep,
)

if autoencoder_config.debug_mode:
    print("DEBUG MODE ENABLED: Overriding training settings.")
    epochs = 2
    batch_size = 2
    nifti_path_list = nifti_path_list[: min(100, len(nifti_path_list))]

config_path = run_output_dir / "run_config.json"
with open(config_path, "w") as f:
    json.dump(asdict(autoencoder_config), f, indent=2)

val_dice_scores = []
val_train_loss = []
val_eval_loss = []

# %%


def train():  # noqa: D103, PLR0915
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and autoencoder_config.device == "cuda"
        else "cpu"
    )
    print(f"Using device: {device}")

    dataset = LesionDataset(nii_paths=nifti_path_list, binarize=False)

    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # use same batch size for validation
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Conv3dAutoencoder(
        input_shape=TARGET_SHAPE_4CHANNEL, latent_dim=N_LATENT_VARIABLES
    ).to(device)

    # Loss, optimizer, and LR on plateau initialisation
    # weight_decay adds L2 regularisation to better handle the large-dimensional model
    criterion = nn.L1Loss(reduction="none")
    optimizer = optim.Adam(
        model.parameters(), lr=autoencoder_config.lr, weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,  # halve the learning rate
        patience=autoencoder_config.patience_reduce_lr,
    )
    # Early stopping and checkpoint setup
    best_val_loss = float("inf")
    patience_counter = 0

    checkpoint_path = run_output_dir / f"{timestamp}_best_autoencoder_full.pt"

    for epoch in range(epochs):
        # ----- TRAIN -----
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch_gpu = batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch_gpu)
            loss = criterion(outputs, batch_gpu)
            weights = 1 + LESION_WEIGHT_MULTIPLIER * batch_gpu
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_gpu.size(0)

        train_loss /= len(train_loader.dataset)

        # ----- VALIDATE -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_gpu = batch.to(device)
                outputs = model(batch_gpu)
                loss = criterion(outputs, batch_gpu)
                weights = 1 + LESION_WEIGHT_MULTIPLIER * batch_gpu
                loss = (loss * weights).mean()
                val_loss += loss.item() * batch_gpu.size(0)

        val_loss /= len(val_loader.dataset)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        val_train_loss.append(train_loss)
        val_eval_loss.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"LR: {current_lr:.6f} "
            f"Output mean: {outputs.mean().item():.4f}"
        )

        # adapt LR on plateau via scheduler
        scheduler.step(val_loss)

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            print("Validation loss improved. Saving model...")
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            print(
                "No improvement. Patience "
                f"{patience_counter}/{autoencoder_config.patience_early_stopping}"
            )

            if patience_counter >= autoencoder_config.patience_early_stopping:
                print("Early stopping triggered!")
                break

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
    metrics_path = run_output_dir / "metrics.json"
    metrics = {
        "train_loss": val_train_loss,
        "val_loss": val_eval_loss,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # save final weights in lightweight model
    lightweight_path = run_output_dir / f"{timestamp}_best_autoencoder_weights.pt"
    torch.save(model.state_dict(), lightweight_path)


if __name__ == "__main__":
    train()

# %%
