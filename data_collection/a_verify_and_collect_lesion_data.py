"""Identify local lesion images and validate data format.

Images with unexpected values (<0, >1, only 0s) are excluded. Consistency of iamge shapes, voxel
sizes, and affines is tested and printed to terminal.

Requirements: Lesion segmentations are stored in LESION_DIR.

Output: csv listing all included images.
"""

# %%
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

LESION_DIR = Path(r"D:\Arbeit_Bern\Data_retrospective\Lesions_raw_cont")
PRECISION_TOLERANCE = 1e-6  # For checks of floating point numbers

# %%
local_nifti = [file for file in LESION_DIR.rglob("*") if file.suffix == ".nii"]
# some images were moved to subfolder "Excluded_normalisation0", indicating bad normalisation to MNI
# space. These should be excluded from the list.
STRING_TO_EXCLUDE = "Excluded_normalisation0"
local_nifti = [p for p in local_nifti if STRING_TO_EXCLUDE not in str(p)]

# %%
# loop through images, validate data, and fetch metadata
image_data_list = []

for path in local_nifti:
    img = nib.load(str(path))
    data_arr = img.get_fdata()

    if np.any(data_arr) is False:
        print(f"Warning: Lesion {path.name} only contains 0 and is excluded.")
        continue
    if np.min(data_arr) < 0.0:
        print(f"Warning: Lesion {path.name} contains value < 0 and is excluded.")
        continue
    if np.max(data_arr) > 1.0 + PRECISION_TOLERANCE:
        print(f"Warning: Lesion {path.name} contains value > 1 and is excluded.")
        continue

    shape = data_arr.shape
    voxel_sizes = img.header.get_zooms()
    voxel_volume_mm3 = voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]
    affine = img.affine

    THRESHOLD_50PERC_PROB = 0.5
    THRESHOLD_20PERC_PROB = 0.2
    # lesion size with binarisation at p>0.2
    lesion_size_p02 = (data_arr > THRESHOLD_20PERC_PROB).sum() * voxel_volume_mm3
    # lesion size with binarisation at p>0.5
    lesion_size_p05 = (data_arr > THRESHOLD_50PERC_PROB).sum() * voxel_volume_mm3

    subject_id = path.stem.replace("wroifinal", "Subject_")

    metadata = {
        "NiftiPath": str(path),
        "SubjectID": subject_id,
        "Shape": shape,
        "VoxelSizes": voxel_sizes,
        "VoxelVolumeML": voxel_volume_mm3,
        "Affine": affine.tolist(),
        "LesionSizeML_p02": lesion_size_p02,
        "LesionSizeML_p05": lesion_size_p05,
    }
    image_data_list.append(metadata)

data_df = pd.DataFrame(image_data_list)

# %%
# verify formatting consistency
if data_df["Shape"].nunique() == 1:
    print("Image shapes are identical.")
else:
    print("WARNING: Image shapes are NOT identical.")

if data_df["VoxelSizes"].nunique() == 1:
    print("Voxel sizes are identical.")
else:
    print("WARNING: Voxel sizes are NOT identical.")

reference = data_df["Affine"].iloc[0]
all_equal = data_df["Affine"].apply(lambda x: np.array_equal(x, reference)).all()

if all_equal:
    print("Affines are identical.")
else:
    print("WARNING: Affines are NOT identical.")

# %%
output_name = Path(__file__).with_suffix(".csv")
data_df.to_csv(output_name, index=False, sep=";")

# %%
