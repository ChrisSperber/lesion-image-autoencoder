"""Train linear autoencoder."""

# %%
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from autoencoder_utils.autoencoder_configs import (
    CHECKPOINT_DIR,
    N_LATENT_VARIABLES,
    TARGET_SHAPE_4CHANNEL,
    config,
)
from autoencoder_utils.autoencoder_dataset import LesionDataset
from autoencoder_utils.models.autoencoder_linear import LinearAutoencoder
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from utils import AutoencoderType, dice_score_autoencoder

AUTOENCODER_TYPE = AutoencoderType.LINEAR_BINARY_INPUT

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
# assign training variables
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
epochs = config.epochs

if AUTOENCODER_TYPE in (
    AutoencoderType.LINEAR_BINARY_INPUT,
    AutoencoderType.LINEAR_CONTINUOUS_INPUT,
):
    batch_size = config.batch_size_linear
elif AUTOENCODER_TYPE in (
    AutoencoderType.DEEP_NONLINEAR_BINARY_INPUT,
    AutoencoderType.DEEP_NONLINEAR_CONTINUOUS_INPUT,
):
    batch_size = config.batch_size_deep
else:
    raise ValueError(f"Invalid autoencoder type {AUTOENCODER_TYPE}")

if config.debug_mode:
    print("DEBUG MODE ENABLED: Overriding training settings.")
    epochs = 2
    batch_size = 2
    nifti_path_list = nifti_path_list[: min(100, len(nifti_path_list))]

with open(f"run_config_{timestamp}.json", "w") as f:
    json.dump(asdict(config), f, indent=2)

# %%


def train():  # noqa: D103
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    dataset = LesionDataset(nii_paths=nifti_path_list, binarize=True)

    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # use same batch size for validation
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LinearAutoencoder(
        input_shape=TARGET_SHAPE_4CHANNEL, latent_dim=N_LATENT_VARIABLES
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Early stopping and checkpoint setup
    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = (
        CHECKPOINT_DIR / f"{timestamp}_best_autoencoder_{AUTOENCODER_TYPE.value}.pt"
    )

    for epoch in range(epochs):
        # ----- TRAIN -----
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch_gpu = batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch_gpu)
            loss = criterion(outputs, batch_gpu)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_gpu.size(0)

        train_loss /= len(train_loader.dataset)

        # ----- VALIDATE -----
        model.eval()
        val_loss = 0.0
        dice_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_gpu = batch.to(device)
                outputs = model(batch_gpu)
                loss = criterion(outputs, batch_gpu)
                val_loss += loss.item() * batch_gpu.size(0)

                # Optional: Calculate Dice score
                dice_total += dice_score_autoencoder(outputs, batch_gpu)

        val_loss /= len(val_loader.dataset)
        dice_avg = dice_total / len(val_loader)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Dice: {dice_avg:.4f} "
            f"LR: {current_lr:.6f}"
        )

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
                f"No improvement. Patience {patience_counter}/{config.patience_early_stopping}"
            )

            if patience_counter >= config.patience_early_stopping:
                print("Early stopping triggered!")
                break

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
