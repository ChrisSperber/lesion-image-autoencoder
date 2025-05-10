"""Utiliy objects for autoencoder training and evaluation."""

from enum import Enum
from pathlib import Path

import torch
from autoencoder_utils.autoencoder_configs import (
    TARGET_SHAPE_4CHANNEL,
)
from autoencoder_utils.models.autoencoder_deep_nonlinear import Conv3dAutoencoder
from autoencoder_utils.models.autoencoder_linear import LinearAutoencoder
from utils import N_LATENT_VARIABLES

AUTOENCODER_OUTPUT_DIR = Path(__file__).parent / "autoencoder_utils" / "outputs"


class AutoencoderType(Enum):
    """Enum to define naming tags for autoencoders."""

    LINEAR_BINARY_INPUT = "linear_binary_input"
    LINEAR_CONTINUOUS_INPUT = "linear_continuous_input"
    DEEP_NONLINEAR_BINARY_INPUT = "deep_nonlinear_binary_input"
    DEEP_NONLINEAR_CONTINUOUS_INPUT = "deep_nonlinear_continuous_input"


def dice_score_autoencoder(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7
):
    """Dice score computation for autoencoder pipeline.

    Args:
        pred: Predicted masks, shape (batch_size, C, D, H, W), values in [0, 1].
        target: Ground truth masks, same shape.
        threshold: Threshold to binarize the predicted masks.
        eps: Small value to prevent division by zero.

    Returns:
        float_: Average Dice score over the batch.

    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum(
        dim=[1, 2, 3, 4]
    )  # Sum across (C, D, H, W)
    union = pred_bin.sum(dim=[1, 2, 3, 4]) + target_bin.sum(dim=[1, 2, 3, 4])

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def get_batch_size_for_type(
    autoencoder_type: AutoencoderType, batch_size_linear: int, batch_size_deep: int
) -> int:
    """Define batch size according to Autoencoder type (linear/deep non-linear).

    Args:
        autoencoder_type (AutoencoderType): Autoencoder type defined by Enum.
        batch_size_linear (int): Batch size if linear autoencoder.
        batch_size_deep (int): Batch size if deep nonlinear autoencoder.

    Raises:
        ValueError: Unknown Autoencoder type.

    Returns:
        int: Batch size

    """
    if autoencoder_type in (
        AutoencoderType.LINEAR_BINARY_INPUT,
        AutoencoderType.LINEAR_CONTINUOUS_INPUT,
    ):
        return batch_size_linear
    elif autoencoder_type in (
        AutoencoderType.DEEP_NONLINEAR_BINARY_INPUT,
        AutoencoderType.DEEP_NONLINEAR_CONTINUOUS_INPUT,
    ):
        return batch_size_deep
    else:
        raise ValueError(f"Invalid autoencoder type {autoencoder_type}")


def find_model_weights_path_for_autoencoder_type(
    autoencoder_type: AutoencoderType,
) -> Path:
    """Find the weights file for an autoencoder type.

    Args:
        autoencoder_type (AutoencoderType): Autoencoder type

    Raises:
        FileNotFoundError: No weights file exists.
        ValueError: More than one weights file exists.

    Returns:
        Path: Path to the weights file of Autoencoder type.

    """
    pt_files = list(AUTOENCODER_OUTPUT_DIR.glob("*best_autoencoder_weights.pt"))
    relevant_weights = [f for f in pt_files if autoencoder_type.value in str(f)]

    if len(relevant_weights) == 0:
        raise FileNotFoundError(f"No weights file for {autoencoder_type.value} found.")
    if len(relevant_weights) > 1:
        raise ValueError(
            f"More than one weights file for {autoencoder_type.value} found."
        )
    return relevant_weights[0]


def load_autoencoder_model_for_type(
    autoencoder_type: AutoencoderType, device: torch.device
) -> torch.nn.Module:
    """Load weighting model for a given Autoencoder type.

    Args:
        autoencoder_type (AutoencoderType): Autoencoder type
        device (torch.device): cpu/cuda

    Raises:
        ValueError: Unknown autoencoder type

    Returns:
        torch.nn.Module: Model weightings.

    """
    weights_path = find_model_weights_path_for_autoencoder_type(autoencoder_type)

    if autoencoder_type in (
        AutoencoderType.LINEAR_BINARY_INPUT,
        AutoencoderType.LINEAR_CONTINUOUS_INPUT,
    ):
        model = LinearAutoencoder(
            input_shape=TARGET_SHAPE_4CHANNEL,
            latent_dim=N_LATENT_VARIABLES,
        )
    elif autoencoder_type in (
        AutoencoderType.DEEP_NONLINEAR_BINARY_INPUT,
        AutoencoderType.DEEP_NONLINEAR_CONTINUOUS_INPUT,
    ):
        model = Conv3dAutoencoder(
            input_shape=TARGET_SHAPE_4CHANNEL,
            latent_dim=N_LATENT_VARIABLES,
        )
    else:
        raise ValueError(f"Unsupported AutoencoderType: {autoencoder_type}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
