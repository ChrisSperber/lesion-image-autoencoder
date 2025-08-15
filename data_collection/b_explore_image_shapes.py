"""Explore lesion sample to find optimal shapes for images in the analysis.

An overlay topography of all lesions is generated and the minimum shape required to retain all
relevant information is assessed. This information is required to decide on optimal cropping/padding
strategies.

Voxels with values <0.2 are ignored to remove noise.

Outputs: Overlay topography of all lesions.
"""

# %%
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

DATA_CSV = "a_verify_and_collect_lesion_data.csv"
THRESHOLD_LESION_BINARISATION = 0.2

# %%
parent_dir = Path(__file__).parent
data_df = pd.read_csv(parent_dir / DATA_CSV, sep=";")

# %%
# load first image to reference shape
reference_image = nib.load(data_df.loc[0, "NiftiPath"])
lesion_overlap_arr = np.zeros(reference_image.shape, dtype=np.uint16)

# %%
# create overlap plot with binarised lesion masks

for _, row in data_df.iterrows():
    lesion_nifti = nib.load(row["NiftiPath"])
    lesion_arr = lesion_nifti.get_fdata()

    lesion_arr_binary = (lesion_arr > THRESHOLD_LESION_BINARISATION).astype(np.uint16)
    lesion_overlap_arr = lesion_overlap_arr + lesion_arr_binary

# %%
# store overlap plot for methods description
affine = reference_image.affine
header = reference_image.header.copy()
header.set_data_dtype(np.uint16)
overlap_nifti = nib.Nifti1Image(lesion_overlap_arr, affine=affine, header=header)

max_overlap_val = np.max(lesion_overlap_arr)
filename = (
    parent_dir / f"overlap_topography_n{len(data_df)}_max{max_overlap_val}.nii.gz"
)

overlap_nifti.to_filename(str(filename))

# %%
# trim overlap plot
x_nonzero = np.any(lesion_overlap_arr, axis=(1, 2))
y_nonzero = np.any(lesion_overlap_arr, axis=(0, 2))
z_nonzero = np.any(lesion_overlap_arr, axis=(0, 1))


x_min, x_max = np.where(x_nonzero)[0][[0, -1]]
y_min, y_max = np.where(y_nonzero)[0][[0, -1]]
z_min, z_max = np.where(z_nonzero)[0][[0, -1]]

# Slice the array to remove all-zero borders
trimmed_overlap_array = lesion_overlap_arr[
    x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1
]
print(f"The shape carrying information is {np.shape(trimmed_overlap_array)}")

# %%
