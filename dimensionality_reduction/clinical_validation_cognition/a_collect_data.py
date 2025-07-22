"""Collect clinical data and create imaging data for new sample.

Collect relevant data (cognitive scores, basic descriptives) of the new sample and create latent
imaging variables for this dataset which was not included in the creation and validation of
autoencoders.
Anonymous data are fetched from local excel tables. Original files are confidential and therefore
not included in the repo.

Creation of latent imaging variables requires reprocessing of the main sample to re-create the
baseline models for transformation into latent space (PCA,NMF,truncated SVD), which were not stored
previously. The model for the autoencoder model was stored and is loaded from disk.

Note: The additional images are only available with binarisation, hence latent space creation from
continuous data is omitted.

Outputs:
    - csv containing cognitive scores for clinical evaluation
    - npz files containing arrays of latent variables
"""

# %%
import sys
from pathlib import Path

# add parent to syspath to import utils
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from autoencoder_utils.autoencoder_configs import autoencoder_config
from autoencoder_utils.autoencoder_utils import (
    AutoencoderType,
    load_autoencoder_model_for_type,
)
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from utils import (
    BINARISATION_THRESHOLD_ORIG_LESION,
    N_LATENT_VARIABLES,
    RNG_SEED,
    load_image_as_5d_tensor,
    load_vectorised_images,
)

DATA_XLS_COGN_SCORES = Path(
    r"D:\Arbeit_Bern\Projekt_Disonnection_stroke_outcome\_obs"
    r"\Revision_annals\R1 analysis specific deficits"
    r"\ZipCognition_updated_cleaned.xlsx"
)

WORD_FLUENCY_COL = "phon_alt_g_r_Roh"
SEL_ATTENTION_COL = "Selective attention (total omissions)"

LESION_DIR_COGNITION = Path(
    r"D:\Arbeit_Bern\Projekt_Disonnection_stroke_outcome\_obs"
    r"\Revision_annals\R1 analysis specific deficits\Lesions_nifti"
)

DATA_CSV_MAIN_SAMPLE = (
    Path(__file__).parents[2]
    / "data_collection"
    / "a_verify_and_collect_lesion_data.csv"
)

LESION_PATH_COLUMN = "NiftiPath"

LESION_DIR_COGNITION_SAMPLE = Path(
    r"D:\Arbeit_Bern\Projekt_Disonnection_stroke_outcome\_obs"
    r"\Revision_annals\R1 analysis specific deficits"
    r"\Lesions_nifti\Lesion_orig_size"
)

UID_COLUMN_COGNITION = "Name Zip"

MAXIMUM_ITERATIONS_NMF = 350  # default 200 gave warning, hence increased

OUTPUT_DIR_COMPRESSED_DATA = "output_compressed_images_cognition_sample"

# %%
# load data
# main sample for re-creation of transformations via pca, svd, nmf
data_main_sample_df = pd.read_csv(DATA_CSV_MAIN_SAMPLE, sep=";")

images_2d_main_sample_arr = load_vectorised_images(
    data_main_sample_df[LESION_PATH_COLUMN].tolist()
)
images_2d_main_sample_arr_binary = (
    images_2d_main_sample_arr > BINARISATION_THRESHOLD_ORIG_LESION
).astype(np.uint8)

# new additional sample with cognition data
data_cognition_sample_df = pd.read_excel(DATA_XLS_COGN_SCORES)

data_cognition_sample_df[LESION_PATH_COLUMN] = data_cognition_sample_df[
    UID_COLUMN_COGNITION
].apply(lambda x: str(LESION_DIR_COGNITION_SAMPLE / f"{x + 1000}wROIfinal.nii"))

# NOTE: The images in the provided folder have the same shape and orientation as the main sample,
# i.e. 79x95x79 in 2x2x2mm³ MNI space
# However, they are already binarised
images_2d_cognition_sample_arr_binary = load_vectorised_images(
    data_cognition_sample_df[LESION_PATH_COLUMN].tolist()
)
# sanity check binary format
if np.isin(images_2d_cognition_sample_arr_binary, [0, 1]).all():
    print("Sanity check passed, all loaded new data are already binary")
else:
    msg = "Sanity check failed, not all loaded new data are binary"
    raise ValueError(msg)

# adapt format to main sample data
images_2d_cognition_sample_arr_binary = images_2d_cognition_sample_arr_binary.astype(
    np.uint8
)

target_score_word_fluency = data_cognition_sample_df[WORD_FLUENCY_COL]
# selective attention is strongly skewed -> deskew via log transformation
target_score_sel_attention = np.log1p(
    data_cognition_sample_df[SEL_ATTENTION_COL].to_numpy()
)

# %%
# re-create latent space transform for main data sample
# standard PCA on binary data
pca = PCA(n_components=N_LATENT_VARIABLES)
images_pca_binary = pca.fit_transform(images_2d_main_sample_arr_binary)

# Truncated singular value decomposition on binary data
svd = TruncatedSVD(n_components=N_LATENT_VARIABLES)
images_svd_binary = svd.fit_transform(images_2d_main_sample_arr_binary)

# non-negative matrix factorisation (NMF) on binary data
nmf_model = NMF(
    n_components=N_LATENT_VARIABLES,
    random_state=RNG_SEED,
    max_iter=MAXIMUM_ITERATIONS_NMF,
)
X_nmf_binary = nmf_model.fit_transform(images_2d_main_sample_arr_binary)

# %%
# apply transformations to new sample data
# Apply the same PCA
images_pca_new = pca.transform(images_2d_cognition_sample_arr_binary)

# Apply the same Truncated SVD
images_svd_new = svd.transform(images_2d_cognition_sample_arr_binary)

# Apply the same NMF
images_nmf_new = nmf_model.transform(images_2d_cognition_sample_arr_binary)

# %%
# apply deep autoencoder model on images
device = autoencoder_config.device
autoencoder_type = AutoencoderType.DEEP_NONLINEAR_BINARY_INPUT

# load model
model_weights = load_autoencoder_model_for_type(autoencoder_type, device)
# Split encoder from the full model for creation of latent variables
encoder = model_weights.encoder if hasattr(model_weights, "encoder") else None

# initialise list to store latent variables per subject
ae_latent_vectors_binary = []

for lesion_path in data_cognition_sample_df[LESION_PATH_COLUMN].tolist():
    lesion_5d_tensor = load_image_as_5d_tensor(lesion_path=lesion_path, mode="binary")

    # pass the image through the entire model (which includes both the encoder and decoder)
    with torch.no_grad():
        lesion_5d_tensor = lesion_5d_tensor.to(device)

        if encoder is not None:
            latent = (
                model_weights.encode_latent(lesion_5d_tensor).squeeze().cpu().numpy()
            )
            ae_latent_vectors_binary.append(latent)

images_deep_ae_new = np.array(ae_latent_vectors_binary)

# %%
# store compressed images
output_base_path = Path(__file__).parent / OUTPUT_DIR_COMPRESSED_DATA
output_base_path.mkdir(exist_ok=True)
filename = output_base_path / "compressed_images_cognition"
np.savez(
    filename,
    images_compressed_pca=images_pca_new,
    images_compressed_svd=images_svd_new,
    images_compressed_nmf=images_nmf_new,
    images_compressed_deep_ae=images_deep_ae_new,
)

# %%
# export csv
cols_to_store = [
    "Age",
    "Sex",
    SEL_ATTENTION_COL,
    WORD_FLUENCY_COL,
    UID_COLUMN_COGNITION,
    LESION_PATH_COLUMN,
]
output_cognition_sample_df = data_cognition_sample_df[cols_to_store]

output_name = Path(__file__).with_suffix(".csv")
output_cognition_sample_df.to_csv(output_name, index=False, sep=";")

# %%
