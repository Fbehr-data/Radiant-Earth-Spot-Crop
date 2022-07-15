# import the needed modules
import os

import numpy as np
import pandas as pd

from average_per_mean_function import feat_engi_date
from cloud_masking_function import cloud_mask
from find_repo_root import get_repo_root
from spectral_indices import cal_spectral_indices, drop_na
from train_test_function import train_test_split_fields

# Get function
root_path = get_repo_root()
data_path = f"{root_path}/data/mean_band_perField_perDate.csv"

# Load data
df = pd.read_csv(data_path)

# ****************************************************************
# Cloudmasking

df_tmp = cloud_mask(df, drop_unknown=True, verbose=True)

print(f"Rows of Original Data: {df.shape[0]}")
print(f"Rows of Without Cloud Data: {df_tmp.shape[0]}")


# ****************************************************************
# Spectral Indices
df_tmp = cal_spectral_indices(df_tmp)

# Drop NA
df_tmp = drop_na(df_tmp)

# ****************************************************************
# Mean per Month + Time-transformation
df_tmp = feat_engi_date(df_tmp)

# Drop April TODO: Make a function out of this ... Not sure
df_tmp_x = df_tmp[[s for s in df_tmp.columns if not "_4" in s]]

# Drop NA
df_tmp = df_tmp_x.dropna()

# Print out Number of rows
print(f"Lost number of fields: {df_tmp_x.shape[0] - df_tmp.shape[0]}")

# Save data
df_tmp.to_csv("data/test_FE.csv", index=False)
# df_tmp.to_csv('data/data_after_FE.csv', index = False)
