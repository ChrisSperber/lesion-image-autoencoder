"""Utiliy objects."""

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
