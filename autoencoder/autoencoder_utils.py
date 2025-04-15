"""Dataset class for loading lesions."""

import nibabel as nib
import numpy as np
import torch

from utils import pad_to_shape

TARGET_SHAPE = (80, 96, 80)


class LesionDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading and preprocessing 3D lesion segmentation maps from NIfTI files.

    This dataset supports minimal on-the-fly preprocessing:
    - Loads probabilistic lesion masks with values in [0, 1]
    - Applies thresholding to remove low-probability values which are potential noise (set to 0 to
      deactivate)
    - Optionally binarizes the lesion map (0 or 1)
    - Pads the volume to a target shape (e.g., (80, 96, 80))
    - Adds a channel dimension for compatibility with 3D CNNs

    Args:
        nii_paths (list of str): Paths to .nii or .nii.gz files containing lesion masks.
        pad_to (tuple of int): Target shape (D, H, W) to pad each image to. Default is (80, 96, 80).
        threshold (float): Probability threshold below which values are set to 0. Default is 0.2.
        binarize (bool): If True, binarizes the lesion map (values set to 0 or 1). Default is False.

    Returns:
        torch.Tensor: 4D tensor of shape (1, D, H, W) with dtype float32, suitable for model input.

    """

    def __init__(
        self, nii_paths, pad_to=TARGET_SHAPE, threshold=0.2, binarize=False
    ):  # noqa: D107
        self.nii_paths = nii_paths
        self.pad_to = pad_to
        self.threshold = threshold
        self.binarize = binarize

    def __len__(self):  # noqa: D105
        return len(self.nii_paths)

    def __getitem__(self, idx):  # noqa: D105
        path = self.nii_paths[idx]
        img = nib.load(path).get_fdata().astype(np.float32)

        img[img < self.threshold] = 0

        if self.binarize:
            img = (img >= self.threshold).astype(np.uint8)

        img = pad_to_shape(img, self.pad_to)

        # Add channel dimension for PyTorch (C, D, H, W)
        img = torch.from_numpy(img).unsqueeze(0)

        return img
