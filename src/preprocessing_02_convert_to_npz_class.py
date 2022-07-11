## Preprocessing 03: Calculation of the mean of the bands # Max Langer # 2022-07-06 ##
## The script is based on the solution of Kiminya for the Zindi: Spot the crop challenge.
## https://github.com/RadiantMLHub/spot-the-crop-challenge/tree/main/2nd%20place%20-%20Kiminya

# import the needed modules
import os, sys, pickle, multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from tqdm.auto import tqdm
import rasterio

class ConversionToNPZ():
  def __init__(self, ROOT_DIR:str) -> None:
    # set the directories
    self.ROOT_DIR = ROOT_DIR
    self.DATA_DIR = f"{self.ROOT_DIR}/data"
    self.IMAGE_DIR = f"{self.DATA_DIR}/images"
    self.BANDS_DIR = f"{self.DATA_DIR}/bands-raw" 
    os.makedirs(self.BANDS_DIR, exist_ok=True)

  def get_bands(self) -> list:
    """ Load the used bands.

    Returns:
        list: List of the used bands.
    """
    bands = pd.read_pickle(f"{self.IMAGE_DIR}/used_bands.pkl")
    bands = bands.used_bands.tolist()
    return bands

  # Function for extracting the pixel information of each tile for each band
  def extract_s2(self, df_tiles:pd.DataFrame) -> pd.DataFrame:
    """ Extracts the pixel information of each tile for each band.
        The pixel information of each field is saved in a npz object.
        The meta data is given back as a pandas data frame. 

    Args:
        df_tiles (pd.DataFrame): Frame with the tiles to be processed.

    Returns:
        pd.DataFrame: Meta data for the tiles and their fields.
    """
    fields = []         # create empty list to catch the field ids
    labels = []         # create empty list to catch the labels
    dates = []          # create empty list to catch the dates for each tile
    tiles = []          # create empty list to catch the tile ids
    
    tile_ids = df_tiles["tile_id"].unique().tolist()
    bands = self.get_bands()

    for tile_id in tqdm(tile_ids):                          # iterate through each tile id
        df_tile = df_tiles[df_tiles["tile_id"]==tile_id]    # load a data frame with the data of the current tile id
        tile_dates = sorted(df_tile[df_tile["satellite_platform"]=="s2"]["date"].unique())    # sort data by date
        
        ARR = {}                                          # create dictionary to catch all the band information for all dates of the current tile
        for band in bands:                                # iterate through the bands we chose
          band_arr = []                                   # create empty list to catch the band data for each date
          for date in tile_dates:                         # iterate through the dates for the current tile id 
            src = rasterio.open(df_tile[(df_tile["date"]==date) & (df_tile["asset"]==band)]["file_path"].values[0])
            band_arr.append(src.read(1))                  # open the band data (pixel) for the current band of the current tile and current date
          ARR[band] = np.array(band_arr,dtype="float32")  # add the band data to the dictionary under the current band name
          
        multi_band_arr = np.stack(list(ARR.values())).astype(np.float32)    # reformats the dictionary values (arrays of the bands) to a stacked array
        multi_band_arr = multi_band_arr.transpose(2,3,0,1)                  # reformats the dictionary values to the shape: width, height, bands, dates
        label_src = rasterio.open(df_tile[df_tile["asset"]=="labels"]["file_path"].values[0])
        label_array = label_src.read(1)                   # reads the labels of the pixels that belong to fields in the tile
        field_src = rasterio.open(df_tile[df_tile["asset"]=="field_ids"]["file_path"].values[0])
        fields_arr = field_src.read(1)                    # reads the field id of the pixels that belong to fields in tile
        
        for field_id in np.unique(fields_arr):            # iterate through all field ids in the current tile
          if field_id==0:                                 # ignore fields with id 0 since these are no fields
            continue
          mask = fields_arr==field_id                     # create a mask of the pixels that belong to the current field id
          field_label = np.unique(label_array[mask])      # use the mask to get the label of the current field id
          field_label = [l for l in field_label if l!=0]  # ignores labels that are 0 since these are no fields
          
          if len(field_label)==1:                         # ignore fields with multiple labels
            field_label = field_label[0]                  # convert the label array to an integer
            patch = multi_band_arr[mask]                  # use the mask to determines which pixels for all the bands and dates belong to the current field id
            np.savez_compressed(f"{self.BANDS_DIR}/{field_id}", patch) # save these pixels of the bands array as np object
            
            labels.append(field_label)                    # add the current field label
            fields.append(field_id)                       # add the current field id
            tiles.append(tile_id)                         # add the current tile id
            dates.append(tile_dates)                      # add the dates which are available for the current tile
    df = pd.DataFrame(dict(field_id=fields,tile_id=tiles,label=labels,dates=dates)) # create a dataframe from the meta data
    return df

  def start_conversion(self):
    # load the data
    print("Loading the image info... \n")
    df_images = pd.read_csv(f"{self.IMAGE_DIR}/images_info_data.csv")
    df_images["date"] = df_images.datetime.astype(np.datetime64)
    bands = self.get_bands()

    # create a sorted dataframe by the tile ids
    tile_ids = sorted(df_images.tile_id.unique())
    print(f"Extracting data from {len(tile_ids)} tiles for bands {bands}")

    # check the number of CPU cores
    num_processes = 4
    print(f"Processesing on : {num_processes} CPUs \n")
    print(f"Start the conversion process ... \n")

    # create a pool of processes equal to the number of cores
    pool = multiprocessing.Pool(num_processes)

    # calculate the number of tiles each core must process
    tiles_per_process = len(tile_ids) / num_processes

    # create the a number of tile id batches equal to the number of cores
    batches = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * tiles_per_process + 1
        end_index = num_process * tiles_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        sublist = tile_ids[start_index - 1:end_index]
        batches.append((sublist,))
        print(f"Task # {num_process} process tiles {len(sublist)}")

    # start the processes and catch the results
    processes = []
    for batch in batches:
      df_batch = df_images[df_images["tile_id"].isin(np.array(batch))]
      processes.append(pool.apply_async(self.extract_s2, args=(df_batch,)))

    # start the processes and catch the results
    results = [p.get() for p in processes]

    # create a data frame from the meta data results and save it as pickle file
    df_meta = pd.concat(results)
    df_meta = df_meta.sort_values(by=["field_id"]).reset_index(drop=True)
    df_meta.to_pickle(f"{self.DATA_DIR}/meta_data_fields_bands.pkl")

    print(f"Training bands saved to {self.BANDS_DIR}")
    print(f"Training meta data saved to {self.DATA_DIR}/meta_data_fields_bands.pkl")


if __name__ == "__main__":
    from find_repo_root import get_repo_root
    ROOT_DIR = get_repo_root()
    conversion = ConversionToNPZ(ROOT_DIR)
    conversion.start_conversion()
