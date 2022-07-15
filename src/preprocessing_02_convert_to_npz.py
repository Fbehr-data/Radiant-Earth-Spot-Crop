## Preprocessing 03: Calculation of the mean of the bands # Max Langer # 2022-07-11 ##
## The script is based on the solution of Kiminya for the Zindi: Spot the crop challenge.
## https://github.com/RadiantMLHub/spot-the-crop-challenge/tree/main/2nd%20place%20-%20Kiminya

import multiprocessing
# import the needed modules
import os
import pickle
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


class ConversionToNPZ:
    """Class to convert the TIF files from the
    satellite data into arrays and save them as NPZ files.
    """

    def __init__(self, ROOT_DIR: str) -> None:
        # set the directories
        self.ROOT_DIR = ROOT_DIR
        self.DATA_DIR = f"{self.ROOT_DIR}/data"
        self.IMAGE_DIR = f"{self.DATA_DIR}/images"
        self.BANDS_DIR = f"{self.DATA_DIR}/bands-raw"
        os.makedirs(self.BANDS_DIR, exist_ok=True)

    def get_bands(self) -> list:
        """Load the used bands.

        Returns:
            list: List of the used bands.
        """
        bands = pd.read_pickle(f"{self.IMAGE_DIR}/used_bands.pkl")
        bands = bands.used_bands.tolist()
        return bands

    # drop zero values for patch
    def drop_zero_values(self, GLCM: np.array) -> np.array:
        """The calculation of the metrics need and integer array, it can not handle NA values
          For this reason we have to delete in the GLCM all entries to the zero

        Args:
            GLCM (np.array): Grey level occurence matrix

        Returns:
            GLCM (np.array): Grey level occurence matrix without values connected to zero
        """
        # change 4-D Array to 2D array
        array_tmp = np.squeeze(GLCM)
        # Drop first row and column
        return array_tmp[1:, 1:, np.newaxis, np.newaxis]

    def calc_texture_index(self, patch_RGB: np.array, metric: str):
        """calculates the given texture index

        Args:
            patch_RGB (np.array): patch of one field (combined visual indices)
            metric (str): Give string to calculate the regarding metric-> possible strings  'correlation', 'homogeneity', 'contrast'

        Returns:
            float: returns the texture index of the given field
        """
        # define scaler for the the field
        minmax_scale = MinMaxScaler(feature_range=(0, 8))
        # apply scaler to the field --> Standard to 8 bit, thus reducing noise
        x_scale = minmax_scale.fit_transform(patch_RGB)
        # change to int for graycomatrix
        x_int = x_scale.astype("int")
        # Calculate the GLCM for distances 1,5 and 10 pixels
        GLCM = graycomatrix(x_int, distances=[5], angles=[0], levels=9, symmetric=True)
        # Drop zero values because we are not interested of pixels outside the field
        GLCM_wo_zeros = self.drop_zero_values(GLCM)
        # Calculate texture metric
        return np.unique(graycoprops(GLCM_wo_zeros, metric))[0]

    # Function for extracting the pixel information of each tile for each band
    def extract_s2(self, df_tiles: pd.DataFrame) -> pd.DataFrame:
        """Extracts the pixel information of each tile for each band.
            The pixel information of each field is saved in a npz object.
            The meta data is given back as a pandas data frame.

        Args:
            df_tiles (pd.DataFrame): Frame with the tiles to be processed.

        Returns:
            pd.DataFrame: Meta data for the tiles and their fields.
        """
        # create empty list to catch the results
        fields = []
        labels = []
        dates = []
        tiles = []
        field_size = []
        list_correlation = []
        list_homogeneity = []
        list_contrast = []

        tile_ids = df_tiles["tile_id"].unique().tolist()
        bands = self.get_bands()

        # iterate through each tile id
        for tile_id in tqdm(tile_ids):
            # load a data frame with the data of the current tile id
            df_tile = df_tiles[df_tiles["tile_id"] == tile_id]
            tile_dates = sorted(
                df_tile[df_tile["satellite_platform"] == "s2"]["date"].unique()
            )

            # create dictionary to catch all the band information for all dates of the current tile
            ARR = {}
            # iterate through the bands we chose
            for band in bands:
                # create empty list to catch the band data for each date
                band_arr = []
                # iterate through the dates for the current tile id
                for date in tile_dates:
                    src = rasterio.open(
                        df_tile[(df_tile["date"] == date) & (df_tile["asset"] == band)][
                            "file_path"
                        ].values[0]
                    )
                    # open the band data (pixel) for the current band of the current tile and current date
                    band_arr.append(src.read(1))
                # add the band data to the dictionary under the current band name
                ARR[band] = np.array(band_arr, dtype="float32")

            # reformats the dictionary values (arrays of the bands) to a stacked array
            multi_band_arr = np.stack(list(ARR.values())).astype(np.float32)
            # reformats the dictionary values to the shape: width, height, bands, dates
            multi_band_arr = multi_band_arr.transpose(2, 3, 0, 1)
            label_src = rasterio.open(
                df_tile[df_tile["asset"] == "labels"]["file_path"].values[0]
            )
            # reads the labels of the pixels that belong to fields in the tile
            label_array = label_src.read(1)  
            # reads the labels of the pixels that belong to fields in the tile
            field_src = rasterio.open(
                df_tile[df_tile["asset"] == "field_ids"]["file_path"].values[0]
            )
            # reads the field id of the pixels that belong to fields in tile
            fields_arr = field_src.read(1)

            # iterate through all field ids in the current tile
            for field_id in np.unique(fields_arr):
                # ignore fields with id 0 since these are no fields
                if field_id == 0:
                    continue
                # create a mask of the pixels that belong to the current field id
                mask2D = fields_arr == field_id
                # use the mask to get the label of the current field id
                field_label = np.unique(label_array[mask2D])
                # ignores labels that are 0 since these are no fields
                field_label = [l for l in field_label if l != 0]

                # ignore fields with multiple labels
                if len(field_label) == 1:
                    # convert the label array to an integer
                    field_label = field_label[0]
                    # use the mask to determines which pixels for all the bands and dates belong to the current field id
                    patch = multi_band_arr[mask2D]
                    # save these pixels of the bands array as np object
                    np.savez_compressed(f"{self.BANDS_DIR}/{field_id}", patch)

                    # change dimension  from 256 x 256 to 256 x 256 x7 x 72
                    mask4D = np.broadcast_to(
                        mask2D[:, :, np.newaxis, np.newaxis], multi_band_arr.shape
                    )
                    # apply mask to bands
                    patch = np.where(mask4D, multi_band_arr, np.nan)
                    # choose band 2,3,4 --> Sum UP
                    patch_RGB = (
                        patch[:, :, 1, :] + patch[:, :, 2, :] + patch[:, :, 3, :]
                    )
                    # transpose dimension for vectorization
                    patch_RGB_T = patch_RGB.transpose(2, 0, 1)

                    # add texture metrics
                    list_correlation.append(
                        [self.calc_texture_index(x, "correlation") for x in patch_RGB_T]
                    )
                    list_homogeneity.append(
                        [self.calc_texture_index(x, "homogeneity") for x in patch_RGB_T]
                    )
                    list_contrast.append(
                        [self.calc_texture_index(x, "contrast") for x in patch_RGB_T]
                    )
                    # add the results to the container lists
                    labels.append(field_label)
                    fields.append(field_id)
                    field_size.append(np.count_nonzero(mask2D))
                    tiles.append(tile_id)
                    dates.append(tile_dates)

        # create a dataframe from the meta data
        df = pd.DataFrame(
            dict(
                field_id=fields,
                tile_id=tiles,
                label=labels,
                field_size=field_size,
                dates=dates,
                correlation=list_correlation,
                homogeneity=list_homogeneity,
                contrast=list_contrast,
            )
        )

        return df

    def start_conversion(self):
        """Starts the conversion process."""
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
            sublist = tile_ids[start_index - 1 : end_index]
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


def main(ROOT_DIR: str):
    ROOT_DIR = get_repo_root()
    conversion = ConversionToNPZ(ROOT_DIR)
    conversion.start_conversion()


if __name__ == "__main__":
    from find_repo_root import get_repo_root

    ROOT_DIR = get_repo_root()
    main(ROOT_DIR)
