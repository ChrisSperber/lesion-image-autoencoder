"""Evaluate compression/reconstruction performance of autoencoders.

The analysis mirrors the logic in simple_feature_reduction.py applied to PCA, NMF, and truncated
SVD - all images are compressed, reconstructed, and the reconstruction error is evaluated witht the
same functions.

Requirements:
    - autoencoders for binary/continuous data with linear/deep non-linear nets (i.e. 2x2) were
        succesfully trained and weights were stored

Outputs:
    - a csv with the reconstruction loss for each lesion and autoencoder.
"""

# %%
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from autoencoder_utils.autoencoder_configs import autoencoder_config
from autoencoder_utils.autoencoder_utils import (
    AutoencoderType,
    load_autoencoder_model_for_type,
)
from utils import (
    BINARISATION_THRESHOLD_ORIG_LESION,
    compute_reconstruction_error,
    load_image_as_5d_tensor,
    unpad_to_shape,
)

DATA_COLLECTION_DIR = "data_collection"
DATA_CSV = "a_verify_and_collect_lesion_data.csv"
LESION_PATH_COLUMN = "NiftiPath"

SUBJECT_ID = "SubjectID"

# to apply continuous methods on binary data, reconstructed data are binarised at 0.5
BINARISATION_THRESHOLD_OUTPUTS = 0.5

device = autoencoder_config.device


# %%
# load nifti paths
data_df = pd.read_csv(
    Path(__file__).parents[1] / DATA_COLLECTION_DIR / DATA_CSV, sep=";"
)
# initialise results df
results_df = pd.DataFrame({SUBJECT_ID: data_df[SUBJECT_ID].values})


for autoencoder_type in AutoencoderType:
    reconstruction_errors = []

    # set data mode binary/continuous
    if autoencoder_type in (
        AutoencoderType.LINEAR_BINARY_INPUT,
        AutoencoderType.DEEP_NONLINEAR_BINARY_INPUT,
    ):
        mode = "binary"
    elif autoencoder_type in (
        AutoencoderType.LINEAR_CONTINUOUS_INPUT,
        AutoencoderType.DEEP_NONLINEAR_CONTINUOUS_INPUT,
    ):
        mode = "continuous"
    else:
        raise ValueError(f"Unsupported AutoencoderType: {autoencoder_type}")

    # load model
    model_weights = load_autoencoder_model_for_type(autoencoder_type, device)

    for lesion_path in data_df[LESION_PATH_COLUMN].tolist():
        original_nifti: nib.nifti1.Nifti1Image = nib.load(lesion_path)
        original_img = original_nifti.get_fdata(dtype=np.float32)
        if mode == "binary":
            original_img = original_img > BINARISATION_THRESHOLD_ORIG_LESION

        original_img_shape = original_img.shape

        lesion_5d_tensor = load_image_as_5d_tensor(lesion_path=lesion_path, mode=mode)

        # pass the image through the entire model (which includes both the encoder and decoder)
        with torch.no_grad():
            lesion_5d_tensor = lesion_5d_tensor.to(device)
            reconstructed_tensor = model_weights(lesion_5d_tensor)

        reconstructed_img = unpad_to_shape(
            reconstructed_tensor.cpu().squeeze().numpy(),
            original_shape=original_img_shape,
        )
        recon_error = compute_reconstruction_error(
            original=original_img,
            reconstructed=reconstructed_img,
            mode=mode,
        )
        reconstruction_errors.append(recon_error)

    results_df[autoencoder_type.value] = reconstruction_errors

# %%
# store results

output_name = Path(__file__).with_suffix(".csv")
results_df.to_csv(output_name, index=False, sep=";")
# %%
