"""Utiliy objects."""

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import torch
from autoencoder_utils.autoencoder_configs import TARGET_SHAPE

# cutoff to binarise original lesions. WARNING: Only applies to original segmentations to compute
# the ground truth
BINARISATION_THRESHOLD_ORIG_LESION = 0.2

RNG_SEED = 9001

# set the number of latent variables to 64 for optimal computation time with autoencoder
N_LATENT_VARIABLES: int = 64


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


def load_vectorised_images(
    lesion_path_list: list[str],
    noise_threshold: float = BINARISATION_THRESHOLD_ORIG_LESION,
) -> np.ndarray:
    """Load lesion segmentations from NIfTI files and return a 2D array.

    Each image is flattened into a 1D vector (row), resulting in a 2D matrix
    of shape (n_subjects, n_voxels).

    Args:
        lesion_path_list: List of file paths to 3D lesion NIfTI images. All images must have the
            same shape.
        noise_threshold: All values below this threshold are set to 0, mirroring the procedures in
            the autoencoder dataset.

    Returns:
        2D NumPy array of shape (n_images, n_voxels).

    """
    images = []
    for path in lesion_path_list:
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)

        # Threshold small values to zero
        data[data < noise_threshold] = 0.0

        images.append(data.ravel())
    return np.stack(images)


def load_image_as_5d_tensor(
    lesion_path: Path, mode: Literal["continuous", "binary"]
) -> torch.Tensor:
    """Load a local lesion nifti as a 5d tensor.

    The tensor includes singleton batch and 4th channel dimensions and has size (1, 1, D, H, W).

    Args:
        lesion_path (Path): Path to a local lesion nifti.
        mode (Literal): Format of tensor (binarised or continuous)

    Returns:
        torch.Tensor: (1, 1, D, H, W)-tensor

    """
    nifti: nib.nifti1.Nifti1Image = nib.load(lesion_path)
    if mode == "binary":
        img_arr = (
            nifti.get_fdata(dtype=np.float32) > BINARISATION_THRESHOLD_ORIG_LESION
        ).astype(np.float32)
    elif mode == "continuous":
        img_arr = nifti.get_fdata(dtype=np.float32)
    else:
        raise ValueError("Invalid mode {mode}")

    img_padded = pad_to_shape(img_arr, target_shape=TARGET_SHAPE)
    return torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0)


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
