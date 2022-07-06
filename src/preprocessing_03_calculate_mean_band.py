## Preprocessing 03: Calculation of the mean of the bands # Max Langer # 2022-07-06 ##

# import the needed modules
# import the needed modules
import numpy as np
import pandas as pd
from scipy import stats
import os, random, pickle, time, glob, multiprocessing
from tqdm.auto import tqdm
from collections import OrderedDict

# import own modules from the scr folder
import sys
sys.path.append('../src/')
from preprocessing_functions import get_clm, calculate_band_mode

# set the directory and the chunks in which the larger fields are splitted 
DATA_DIR = './data'
DIR_BANDS = f'{DATA_DIR}/bands-raw/' 

# load the data frame and add the path information of the npz objects for each field to the data frame
df = pd.read_pickle(f'{DATA_DIR}/meta_data_fields_bands.pkl')
df['path'] = DIR_BANDS+df.field_id.astype(str)+'.npz'

# extract the field data from the npz files 
# and calculate the mean of each field for each band on each date
field_ids = []
labels = []
dates = []
features = []

print("Calculation of the mean for each band of each field on each date:")
for _,row in tqdm(df.iterrows(), total=len(df)):
    bands = np.load(row.path)['arr_0']
    n = bands.shape[0]              # save the number of bands 
    n_dates = bands.shape[2]        # save the number of dates 

    bands, cloud = get_clm(bands)                       # get the cloud mask out of the bands and update
    cloud_mode = calculate_band_mode(cloud)             # calculate the mode over each pixel and for all dates for the cloud mask
    mean = np.mean(bands,axis=0)                        # calculate the mean over each pixel for the band and dates
    feature = mean.transpose(1,0)                       # switch the bands and dates
    feature = np.concatenate((feature, cloud_mode), axis=1)     # add the mode of the cloud mask back into the features
    features.append(feature)                            # add the features of each field to the features list
           
    field_id = np.repeat(row.field_id,feature.shape[0]) # get an array of the field ids, of the same size as the date array of the current feature
    field_ids.append(field_id)                          # add the field ids array to the field ids list
    label = np.repeat(row.label,feature.shape[0])       # get an array of the labels, of the same size as the date array of the current feature
    labels.append(label)                                # add the label array to the labels list
    date = [str(d)[:10] for d in row.dates]             # goes through the dates in each row and saves them to a list without the time [-> [:10]]
    date = np.array(date)                               # convert the date list to an array
    dates.append(date)                                  # add the date array to the dates

# put all of the list information into an array
all_features = np.concatenate(features)
all_field_ids = np.concatenate(field_ids)
all_labels = np.concatenate(labels)
all_dates = np.concatenate(dates)

# put all different information into one data frame
cols = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'CLM']
df_data = pd.DataFrame(all_features,columns=cols)
df_data.insert(0,'field_id',all_field_ids)
df_data.insert(1,'date',all_dates)
df_data.insert(2,'label',all_labels)

# save the data frame as CSV file
print(f"Saving the data into {DATA_DIR}/mean_band_perField_perDate.csv")
df_data.to_csv(f'{DATA_DIR}/mean_band_perField_perDate.csv', index=False)
