"""Utiliy objects."""

from typing import Literal

import nibabel as nib
import numpy as np

BINARISATION_THRESHOLD = 0.2  # global defining a cutoff to binarise original lesions
RNG_SEED = 9001


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

    Measure is chosen based on mode either as mean absolute error (continuous) or 1 - Dice (binary).

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
        # Ensure binary (exactly 0 or 1)
        if not (
            np.isin(original, [0, 1]).all() and np.isin(reconstructed, [0, 1]).all()
        ):
            raise ValueError(
                "In 'binary' mode, both inputs must contain only 0 and 1 values."
            )

        intersection = np.sum(original * reconstructed)
        total = np.sum(original) + np.sum(reconstructed)

        if total == 0:
            # Both empty masks — define Dice as 1 (i.e., perfect match), so error is 0
            return 0.0

        dice_score = 2 * intersection / total
        return 1.0 - dice_score

    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'continuous' or 'binary'.")
