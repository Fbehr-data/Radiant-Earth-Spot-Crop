# Import the needed modules
import os, sys, datetime
import numpy as np, pandas as pd
import rasterio
from pathlib import Path
from radiant_mlhub.client import _download as download_file
from collections import OrderedDict
from tqdm.auto import tqdm

# Set the directories
OUTPUT_DIR = '../data'
OUTPUT_DIR = f'{OUTPUT_DIR}/train'
os.makedirs(OUTPUT_DIR,exist_ok=True)
OUTPUT_DIR_BANDS = f'{OUTPUT_DIR}/bands-raw' 
os.makedirs(OUTPUT_DIR_BANDS,exist_ok=True)
DOWNLOAD_S2 = OrderedDict({
    'B01': False,
    'B02': True, #Blue
    'B03': True, #Green
    'B04': True, #Red
    'B05': False,
    'B06': False,
    'B07': False,
    'B08': True, #NIR
    'B8A': False, #NIR2
    'B09': False,
    'B11': True, #SWIR1
    'B12': True, #SWIR2
    'CLM': True
})

# Load the data
df_train = pd.read_csv(f'{OUTPUT_DIR}/train_data.csv')        # load the training data csv
df_train['date'] = df_train.datetime.astype(np.datetime64)    # convert the datetime to np datetime object
bands = [k for k,v in DOWNLOAD_S2.items() if v==True]         # create a list of the bands we chose

# Function for extracting all the 
def extract_s2(tile_ids):
  """ Extracts the pixel information of each tile for each band.
      The pixel information of each field is safed in a npz object.
      The meta data is given back as a pandas data frame. 

  Args:
      tile_ids (list): List of tile ids to be processed.

  Returns:
      pandas data frame: Meta data for the tiles and their fields.
  """
  fields = []         # create empty list to catch the field ids
  labels = []         # create empty list to catch the labels
  dates = []          # create empty list to catch the dates for each tile
  tiles = []          # create empty list to catch the tile ids
  
  for tile_id in tqdm(tile_ids):                          # iterate through each tile id
      df_tile = df_train[df_train['tile_id']==tile_id]    # load a data frame with the data of the current tile id
      tile_dates = sorted(df_tile[df_tile['satellite_platform']=='s2']['date'].unique())    # sort data by date
      
      ARR = {}                                          # create dictionary to catch all the band information for all dates of the current tile
      for band in bands:                                # iterate through the bands we chose
        band_arr = []                                   # create empty list to catch the band data for each date
        for date in tile_dates:                         # iterate through the dates for the current tile id 
          src = rasterio.open(df_tile[(df_tile['date']==date) & (df_tile['asset']==band)]['file_path'].values[0])
          band_arr.append(src.read(1))                  # open the band data (pixel) for the current band of the current tile and current date
        ARR[band] = np.array(band_arr,dtype='float32')  # add the band data to the dictionary under the current band name
        
      multi_band_arr = np.stack(list(ARR.values())).astype(np.float32)    # reformats the dictionary values (arrays of the bands) to a stacked array
      multi_band_arr = multi_band_arr.transpose(2,3,0,1)                  # reformats the dictionary values to the shape: width, height, bands, dates
      label_src = rasterio.open(df_tile[df_tile['asset']=='labels']['file_path'].values[0])
      label_array = label_src.read(1)                   # reads the labels of the pixels that belong to fields in the tile
      field_src = rasterio.open(df_tile[df_tile['asset']=='field_ids']['file_path'].values[0])
      fields_arr = field_src.read(1)                    # reads the field id of the pixels that belong to fields in tile
      
      for field_id in np.unique(fields_arr):            # iterate through all field ids in the current tile
        if field_id==0:                                 # ignore fields with id 0 since these are no fields
          continue
        mask = fields_arr==field_id                     # create a mask of the pixels that belong to the current field id
        field_label = np.unique(label_array[mask])      # use the mask to get the label of the current field id
        field_label = [l for l in field_label if l!=0]  # ignores labels that are 0 since these are no fields
        
        if len(field_label)==1:                         # ignore fields with multiple labels
          field_label = field_label[0]                  # convert the label array to an integer
          patch = multi_band_arr[mask]                  # use the mask to determines which pixels of the bands belong to the current field id
          np.savez_compressed(f"{OUTPUT_DIR_BANDS}/{field_id}", patch) # save these pixels of the bands array as np object
          
          labels.append(field_label)                    # add the current field label
          fields.append(field_id)                       # add the current field id
          tiles.append(tile_id)                         # add the current tile id
          dates.append(tile_dates)                      # add the dates which are available for the current tile
  df = pd.DataFrame(dict(field_id=fields,tile_id=tiles,label=labels,dates=dates)) # create a dataframe from the meta data
  return df