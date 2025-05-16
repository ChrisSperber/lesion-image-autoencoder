"""Collect clinical and demographic data.

Anonymous clinical and demographic data are fetched from local excel tables. Original files
are confidential and therefore not included in the the repo.
The output with NIHSS24h exludes subjects that have no NIHSS 24h measure available.


Requirements:
    - data_collection/a_verify_and_collect_lesion_data.py

Outputs:
    - json containing basic demographic descriptive statistics on the full sample
    - csv containing NIHSS 24h scores for clinical evaluation of compressed imaging markers
"""

# %%

from pathlib import Path

import pandas as pd

DATA_CSV = (
    Path(__file__).parents[1]
    / "data_collection"
    / "a_verify_and_collect_lesion_data.csv"
)
DEMOGRAPHICS_XLS = Path(
    r"D:\Arbeit_Bern\Projekt CR Interaction Mapping\Data\Data_full_02_2024_CR.xlsx"
)

SUBJECT_ID = "SubjectID"
LESION_LATERALITY = "LesionLaterality"
LESION_VOLUME_ML = "LesionSizeML_p02"

# relevant columns in demographics xls
CASE_ID = "CaseIDmod"  # purely numeric with +100.000
AGE = "Age"  # float
SEX = "Sex"  # string Male/Female
NIHSS_ADMISSION = "NIHSS_Admission"
NIHSS_24H = "NIHSS_24h"
THROMBOLYSIS = "Thrombolysis"
MECHANICAL_THROMBECTOMY = "Thrombectomy"
ACUTE_INTERVENTION = "Intervention"

colname_mapping = {
    "NIHonadmission": NIHSS_ADMISSION,
    "NIHSStwentyfourh": NIHSS_24H,
    "IVTwithrtPA": THROMBOLYSIS,
    "mt": MECHANICAL_THROMBECTOMY,
}

COLS_TO_MERGE = [
    AGE,
    SEX,
    NIHSS_ADMISSION,
    NIHSS_24H,
    THROMBOLYSIS,
    MECHANICAL_THROMBECTOMY,
]

# %%
# load & adapt data
data_df = pd.read_csv(DATA_CSV, sep=";")

demographics_df = pd.read_excel(DEMOGRAPHICS_XLS)
demographics_df.rename(columns=colname_mapping, inplace=True)
demographics_df[SUBJECT_ID] = demographics_df[CASE_ID].apply(
    lambda x: f"Subject_{int(x) % 100000:05d}"
)
demographics_df[MECHANICAL_THROMBECTOMY] = demographics_df[MECHANICAL_THROMBECTOMY].map(
    {1.0: "yes", 0.0: "no"}
)
demographics_df[THROMBOLYSIS] = demographics_df[THROMBOLYSIS].replace(
    "started before admission", "yes"
)

data_df = data_df.merge(
    demographics_df[[SUBJECT_ID] + COLS_TO_MERGE],
    on=SUBJECT_ID,
    how="left",
    validate="one_to_one",
)

data_df[ACUTE_INTERVENTION] = data_df[[THROMBOLYSIS, MECHANICAL_THROMBECTOMY]].apply(
    lambda row: "yes" if "yes" in row.values else "no", axis=1
)

# %%
# generate output md file with descriptives
# Define your variables
categorical_vars = [
    SEX,
    LESION_LATERALITY,
    ACUTE_INTERVENTION,
    MECHANICAL_THROMBECTOMY,
    THROMBOLYSIS,
]
continuous_vars = [AGE, LESION_VOLUME_ML, NIHSS_ADMISSION, NIHSS_24H]

# Create output lines
lines = []

lines.append("# Descriptive Statistics\n")
lines.append(f"Total number of subjects: {len(data_df)}\n")

# Categorical summaries
lines.append("## Categorical Variables\n")
for col in categorical_vars:
    lines.append(f"### {col}")
    value_counts = data_df[col].value_counts(dropna=True)
    for val, count in value_counts.items():
        lines.append(f"- {val}: {count}")
    lines.append("")

# Continuous summaries
lines.append("## Continuous Variables\n")
for col in continuous_vars:
    series = data_df[col].dropna()
    lines.append(f"### {col}")
    lines.append(f"- Mean: {series.mean():.2f}")
    lines.append(f"- SD: {series.std():.2f}")
    lines.append(f"- Median: {series.median():.2f}")
    lines.append(f"- Min: {series.min():.2f}")
    lines.append(f"- Max: {series.max():.2f}")
    lines.append("")

# Write to file
filename = Path(__file__)
filename = filename.stem + "_descriptive_statistics.md"
with open(filename, "w") as f:
    f.write("\n".join(lines))

# %%
# extract and store dataset with valid NIHSS 24h scores
data_temp = data_df[data_df[NIHSS_24H].notna()]
data_nihss = data_temp[[SUBJECT_ID, NIHSS_24H]].copy()
output_name = Path(__file__).with_suffix(".csv")
data_nihss.to_csv(output_name, index=False, sep=";")

# %%
