# import the needed modules
import numpy as np
import pandas as pd
import os


from spectral_indices import cal_spectral_indices, drop_na
from train_test_function import train_test_split_fields
from cloud_masking_function import cloud_mask
from average_per_mean_function import feat_engi_date
#from imblearn.over_sampling import SMOTE


# Set Working Directory
#path = os.getcwd()
path = '/Users/felixbehrendt/neuefische/Radiant-Earth-Spot-Crop/'
# Set Workign directory and print
os.chdir(path)
print(f'Current Working directory: {path}')

# Load data
df = pd.read_csv('data/mean_band_perField_perDate.csv')

# ****************************************************************
# Cloudmasking

# df_woCLM = cloud_mask(df)
# print(f'Rows of Original Data: {df.shape[0]}')
# print(f'Rows of Without Cloud Data: {df_woCLM.shape[0]}')

df_tmp= cloud_mask(df, drop_unknown=True, verbose=True)

print(f'Rows of Original Data: {df.shape[0]}')
print(f'Rows of Without Cloud Data: {df_tmp.shape[0]}')


# ****************************************************************
# Spectral Indices
df_tmp = cal_spectral_indices(df_tmp)

# Drop NA
df_tmp = drop_na(df_tmp)

# ****************************************************************
# Mean per Month + Time-transformation
df_tmp = feat_engi_date(df_tmp)

# Drop April TODO: Make a function out of this ... Not sure
df_tmp_x = df_tmp[[s for s in df_tmp.columns if not '_4' in s]]

# Drop NA 
df_tmp = df_tmp_x.dropna()

# Print out Number of rows
print(f'Lost number of fields: {df_tmp_x.shape[0] - df_tmp.shape[0]}')

# Save data
#df_tmp.to_csv('data/data_after_FE.csv', index = False)
