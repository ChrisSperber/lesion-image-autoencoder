"""Utiliy objects."""

import nibabel as nib
import numpy as np


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
