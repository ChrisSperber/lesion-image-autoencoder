"""Compute different variants of PCA.

Continuous data are compressed with standard PCA and truncated SVD, the latter being suited for
highly sprarse data, i.e. it might be better at handling lesion data.
Binary data are compressed with

Outputs:
    A csv with the reconstruction loss for each lesion and method.
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from utils import load_vectorised_images, compute_reconstruction_error

DATA_COLLECTION_DIR = "data_collection"
DATA_CSV = "a_verify_and_collect_lesion_data.csv"
LESION_PATH_COLUMN = "NiftiPath"

SUBJECT_ID = "SubjectID"

# set the total amount of variance the latent variables should explain
PROPORTION_VARIANCE_EXPLAINED = 0.85

# %%
# load data as 2D array, i.e. with vectorised images
data_df = pd.read_csv(
    Path(__file__).parents[1] / DATA_COLLECTION_DIR / DATA_CSV, sep=";"
)

images_2d_arr = load_vectorised_images(data_df[LESION_PATH_COLUMN].tolist())

# %%
# standard PCA
pca = PCA(n_components=PROPORTION_VARIANCE_EXPLAINED)
images_pca = pca.fit_transform(images_2d_arr)

# Reconstruct components back to original space
images_2d_arr_reconstructed_pca = pca.inverse_transform(images_pca)

# evaluate reconstruction
reconstruction_output_pca = []
# Loop through each image and its reconstruction
for _i, (orig, recon) in enumerate(
    zip(images_2d_arr, images_2d_arr_reconstructed_pca, strict=True)
):
    reconstruction_output_pca.append(
        compute_reconstruction_error(orig, recon, mode="continuous")
    )

# %%
total_explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_latents = pca.n_components_
print(f"Number of latent variables is {n_latents}")

# %%
svd = TruncatedSVD(n_components=n_latents)  # use the same number of components here
images_svd = svd.fit_transform(images_2d_arr)

# Reconstruct the original-like array
images_2d_arr_reconstructed_truncsvd = svd.inverse_transform(images_svd)

# evaluate reconstruction
reconstruction_output_truncsvd = []
# Loop through each image and its reconstruction
for _i, (orig, recon) in enumerate(
    zip(images_2d_arr, images_2d_arr_reconstructed_truncsvd, strict=True)
):
    reconstruction_output_truncsvd.append(
        compute_reconstruction_error(orig, recon, mode="continuous")
    )


# %%
# store results
