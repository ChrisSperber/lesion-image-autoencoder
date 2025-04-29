"""Utiliy objects."""

from enum import Enum
from typing import Literal

import nibabel as nib
import numpy as np
import torch

# cutoff to binarise original lesions. WARNING: Only applies to original segmentations to compute
# the ground truth
BINARISATION_THRESHOLD_ORIG_LESION = 0.2

RNG_SEED = 9001


class AutoencoderType(Enum):
    """Enum to define naming tags for autoencoders."""

    LINEAR_BINARY_INPUT = ("_linear_binary_input",)
    LINEAR_CONTINUOUS_INPUT = ("_linear_continuous_input",)
    DEEP_NONLINEAR_BINARY_INPUT = ("_deep_nonlinear_binary_input",)
    DEEP_NONLINEAR_CONTINUOUS_INPUT = ("_deep_nonlinear_continuous_input",)


def pad_to_shape(img: np.ndarray, target_shape: tuple):
    """Pad image with 0s to match target shape.

    Args:
        img: 3D image array
        target_shape: Tuple of target shape

    Returns:
        Images padded with 0s.

    """
    pad_width = []
    for i in range(3):
        total_pad = target_shape[i] - img.shape[i]
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))
    return np.pad(img, pad_width, mode="constant", constant_values=0)


def unpad_to_shape(padded_img: np.ndarray, original_shape: tuple):
    """Remove padding to return image to original shape.

    This function reverses pad_to_shape.

    Args:
        padded_img: 3D image that was padded
        original_shape: Desired output shape (i.e., original before padding)

    Returns:
        Cropped image with original shape.

    """
    slices = []
    for i in range(3):
        total_pad = padded_img.shape[i] - original_shape[i]
        start = total_pad // 2
        end = start + original_shape[i]
        slices.append(slice(start, end))
    return padded_img[tuple(slices)]


def load_vectorised_images(lesion_path_list: list[str]) -> np.ndarray:
    """Load lesion segmentations from NIfTI files and return a 2D array.

    Each image is flattened into a 1D vector (row), resulting in a 2D matrix
    of shape (n_subjects, n_voxels).

    Args:
        lesion_path_list: List of file paths to 3D lesion NIfTI images. All images must have the
            same shape.

    Returns:
        2D NumPy array of shape (n_images, n_voxels).

    """
    images = []
    for path in lesion_path_list:
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        images.append(data.ravel())
    return np.stack(images)


def compute_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mode: Literal["continuous", "binary"] = "continuous",
) -> float:
    """Compute a reconstruction error between two arrays.

    Measure is chosen based on mode either as mean absolute error (continuous) or Dice score
    (binary).

    Args:
        original: The original image (any shape).
        reconstructed: The reconstructed image (same shape as original).
        mode: Either 'continuous' or 'binary'. In binary mode, inputs must be exactly 0 or 1.

    Returns:
        float: Reconstruction error.

    Raises:
        ValueError: If shapes mismatch or binary inputs aren't valid.

    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs. {reconstructed.shape}")

    if mode == "continuous":
        return np.mean(np.abs(original - reconstructed))  # mean absolute error

    elif mode == "binary":
        # Ensure binary format of inputs (exactly 0 or 1)
        if not (
            np.isin(original, [0, 1]).all() and np.isin(reconstructed, [0, 1]).all()
        ):
            raise ValueError(
                "In 'binary' mode, both inputs must contain only 0 and 1 values."
            )

        intersection = np.sum(original * reconstructed)
        total = np.sum(original) + np.sum(reconstructed)

        if np.sum(original) == 0:
            raise ValueError("Original image is an empty zero-array.")

        dice_score = 2 * intersection / total
        return dice_score

    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'continuous' or 'binary'.")


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
