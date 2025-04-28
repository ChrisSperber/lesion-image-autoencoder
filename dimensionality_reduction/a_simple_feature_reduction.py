"""Compute different variants of feature reduction including PCA.

Data are compressed with standard PCA, truncated SVD (which is suited for highly sparse data, i.e.
it might be better at handling lesion data) and non-negative matrix factorisation.
Methods are either used on the continuous segmentation data or binarised segmentation maps that were
binarised to 1|0 at BINARISATION_THRESHOLD, with subsequent binarisation of reconstructed maps at
at p=0.5.
Logistic PCA was tested but was found to be unsuited for the large dataset.

Outputs:
    - a csv with the reconstruction loss for each lesion and method.
    - an npz with the compressed lesion data in a subjects by components 2D format stored in
        a dedicated output folder 'output_compressed_images'
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from utils import (
    BINARISATION_THRESHOLD_ORIG_LESION,
    RNG_SEED,
    compute_reconstruction_error,
    load_vectorised_images,
)

DATA_COLLECTION_DIR = "data_collection"
DATA_CSV = "a_verify_and_collect_lesion_data.csv"
LESION_PATH_COLUMN = "NiftiPath"

SUBJECT_ID = "SubjectID"

# folder to store the compressed data
OUTPUT_DIR_COMPRESSED_DATA = "output_compressed_images"

# set the total amount of variance the latent variables should explain for PCA with continuous data
PROPORTION_VARIANCE_EXPLAINED = 0.75
# to apply continuous methods on binary data, reconstructed data are binarised at 0.5
BINARISATION_THRESHOLD_OUTPUTS = 0.5

MAXIMUM_ITERATIONS_NMF = 300  # default 200 gave warning, hence increased

# %%
# load data as 2D array, i.e. with vectorised images
data_df = pd.read_csv(
    Path(__file__).parents[1] / DATA_COLLECTION_DIR / DATA_CSV, sep=";"
)

images_2d_arr = load_vectorised_images(data_df[LESION_PATH_COLUMN].tolist())
images_2d_arr_binary = (images_2d_arr > BINARISATION_THRESHOLD_ORIG_LESION).astype(
    np.uint8
)

output_base_path = Path(__file__).parent / OUTPUT_DIR_COMPRESSED_DATA
output_base_path.mkdir(exist_ok=True)

#####################
# continuous analyses
#####################

# %%
# standard PCA on continuous data
pca = PCA(n_components=PROPORTION_VARIANCE_EXPLAINED)
images_pca = pca.fit_transform(images_2d_arr)

# Reconstruct components back to original space
images_2d_arr_reconstructed_pca = pca.inverse_transform(images_pca)

# evaluate reconstruction
reconstruction_output_pca = []
# Loop through each image and its reconstruction
reconstruction_output_pca = [
    compute_reconstruction_error(orig, recon, mode="continuous")
    for orig, recon in zip(images_2d_arr, images_2d_arr_reconstructed_pca, strict=True)
]


# %%
total_explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_latents = pca.n_components_
print(
    f"Number of latent variables is {n_latents} explaining {total_explained_variance[-1]}% of "
    "variance in the PCA with continuous data"
)

# %%
# Truncated singular value decomposition on continuous data
svd = TruncatedSVD(n_components=n_latents)  # use the same number of components here
images_svd = svd.fit_transform(images_2d_arr)

# Reconstruct the original-like array
images_2d_arr_reconstructed_truncsvd = svd.inverse_transform(images_svd)

# evaluate reconstruction
reconstruction_output_truncsvd = []
# Loop through each image and its reconstruction
reconstruction_output_truncsvd = [
    compute_reconstruction_error(orig, recon, mode="continuous")
    for orig, recon in zip(
        images_2d_arr, images_2d_arr_reconstructed_truncsvd, strict=True
    )
]

# %%
# non-negative matrix factorisation (NMF) on continuous data
print("Warning, NMF computation may take several minutes up to hours!")
model = NMF(
    n_components=n_latents, random_state=RNG_SEED, max_iter=MAXIMUM_ITERATIONS_NMF
)
X_nmf = model.fit_transform(images_2d_arr)

# Reconstruct to original space and binarise
images_2d_arr_reconstructed_nmf = np.dot(X_nmf, model.components_)

# evaluate reconstruction
reconstruction_output_nmf = []
# Loop through each image and its reconstruction
reconstruction_output_nmf = [
    compute_reconstruction_error(orig, recon, mode="continuous")
    for orig, recon in zip(images_2d_arr, images_2d_arr_reconstructed_nmf, strict=True)
]

#####################
# binary analyses
#####################

# %%
# logistic PCA for binary data
# logistic PCA struggled with the size of the data and, in the end, suffered from numerical
# instabiltiy or exploding gradients

# %%
# standard PCA on binary data
pca = PCA(n_components=n_latents)
images_pca_binary = pca.fit_transform(images_2d_arr_binary)
# Reconstruct components back to original space and binarise
images_2d_arr_reconstructed_pca_binary = (
    pca.inverse_transform(images_pca_binary) > BINARISATION_THRESHOLD_OUTPUTS
).astype(np.uint8)

# evaluate reconstruction
reconstruction_output_pca_binary = []
# Loop through each image and its reconstruction
reconstruction_output_pca_binary = [
    compute_reconstruction_error(orig, recon, mode="binary")
    for orig, recon in zip(
        images_2d_arr_binary, images_2d_arr_reconstructed_pca_binary, strict=True
    )
]

# %%
# Truncated singular value decomposition on binary data
svd = TruncatedSVD(n_components=n_latents)  # use the same number of components here
images_svd_binary = svd.fit_transform(images_2d_arr_binary)

# Reconstruct the original-like array
images_2d_arr_reconstructed_truncsvd_binary = (
    svd.inverse_transform(images_svd_binary) > BINARISATION_THRESHOLD_OUTPUTS
).astype(np.uint8)

# evaluate reconstruction
reconstruction_output_truncsvd_binary = []
# Loop through each image and its reconstruction
reconstruction_output_truncsvd_binary = [
    compute_reconstruction_error(orig, recon, mode="binary")
    for orig, recon in zip(
        images_2d_arr_binary, images_2d_arr_reconstructed_truncsvd_binary, strict=True
    )
]

# %%
# non-negative matrix factorisation (NMF) on binary data
model = NMF(
    n_components=n_latents, random_state=RNG_SEED, max_iter=MAXIMUM_ITERATIONS_NMF
)
X_nmf_binary = model.fit_transform(images_2d_arr_binary)
# Reconstruct to original space and binarise
images_reconstructed_nmf_continuous = np.dot(X_nmf_binary, model.components_)
images_2d_arr_reconstructed_nmf_binary = (
    images_reconstructed_nmf_continuous > BINARISATION_THRESHOLD_OUTPUTS
).astype(np.uint8)

# evaluate reconstruction
reconstruction_output_nmf_binary = []
# Loop through each image and its reconstruction
reconstruction_output_nmf_binary = [
    compute_reconstruction_error(orig, recon, mode="binary")
    for orig, recon in zip(
        images_2d_arr_binary, images_2d_arr_reconstructed_nmf_binary, strict=True
    )
]

# %%
# store results
results_df = pd.DataFrame(
    {
        SUBJECT_ID: data_df[SUBJECT_ID].values,
        "ReconstructionContinuousPCA": reconstruction_output_pca,
        "ReconstructionContinuousTruncSVD": reconstruction_output_truncsvd,
        "ReconstructionContinuousNMF": reconstruction_output_nmf,
        "ReconstructionBinaryPCA": reconstruction_output_pca_binary,
        "ReconstructionBinaryTruncSVD": reconstruction_output_truncsvd_binary,
        "ReconstructionBinaryNMF": reconstruction_output_nmf_binary,
    }
)

filename = output_base_path / "compressed_images_pca_svd_nmf"
np.savez(
    filename,
    images_pca_cont=images_pca,
    images_svd_cont=images_svd,
    images_nmf_cont=X_nmf,
    images_pca_binary=images_pca_binary,
    images_svd_binary=images_svd_binary,
    images_nmf_binary=X_nmf_binary,
)

output_name = Path(__file__).with_suffix(".csv")
results_df.to_csv(output_name, index=False, sep=";")

# %%
