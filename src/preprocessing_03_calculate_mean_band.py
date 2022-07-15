## Preprocessing 03: Calculation of the mean of the bands # Max Langer # 2022-07-11 ##
## The script is based on the solution of Kiminya for the Zindi: Spot the crop challenge.
## https://github.com/RadiantMLHub/spot-the-crop-challenge/tree/main/2nd%20place%20-%20Kiminya

# import the needed modules
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm


class CalculateMeanPerBand:
    """Class to calculate the mean per band
    for each field for each date.
    """

    def __init__(self, ROOT_DIR: str) -> None:
        # set the directory and the chunks in which the larger fields are splitted
        self.ROOT_DIR = ROOT_DIR
        self.DATA_DIR = f"{self.ROOT_DIR}/data"
        self.BANDS_DIR = f"{self.DATA_DIR}/bands-raw/"
        self.IMAGE_DIR = f"{self.DATA_DIR}/images"

    def get_bands(self) -> list:
        """Load the used bands.

        Returns:
            list: List of the used bands.
        """
        bands = pd.read_pickle(f"{self.IMAGE_DIR}/used_bands.pkl")
        bands = bands.used_bands.tolist()
        return bands

    def get_clm(self, bands: np.array):
        """Extracts the cloud mask band from an array of bands.

        Args:
            bands (np.array): Array of pixel of the current field for each band and date.

        Returns:
            bands (np.array): Array without the cloud mask band.
            cloud (np.array): Array of the cloud mask band.
        """
        # switch the pixel and bands
        bands_T = bands.transpose(1, 0, 2)
        # get the last band, which is the cloud mask
        cloud = np.expand_dims(bands_T[len(bands_T) - 1], axis=0)
        # remove the cloud mask band from the array
        bands_T = bands_T[0 : len(bands_T) - 1]
        # switch bands and pixel
        bands = bands_T.transpose(1, 0, 2)
        # switch bands and pixel
        cloud = cloud.transpose(1, 0, 2)
        return bands, cloud

    def calculate_band_mode(self, band: np.array):
        """Calculates the mode for a given band.

        Args:
            band (np.array): Array of pixel of the current field for one band and each date.

        Returns:
            np.array: Mode over all pixel for the band for each date.
        """
        # calculate the mode
        mode = stats.mode(band)
        # reshapes the array into the form (dates, band)
        mode = np.squeeze(mode[0], axis=0).transpose(1, 0)
        return mode

    def start_calculation(self):
        """Start the calculation process."""
        # load the data frame and add the path information of the npz objects for each field to the data frame
        df = pd.read_pickle(f"{self.DATA_DIR}/meta_data_fields_bands.pkl")
        df["path"] = self.BANDS_DIR + df.field_id.astype(str) + ".npz"

        # extract the field data from the npz files
        # and calculate the mean of each field for each band on each date
        field_ids = []
        field_sizes = []
        labels = []
        dates = []
        features = []
        tile_ids = []

        # empty data for texture metrics
        list_correlation = []
        list_homogeneity = []
        list_contrast = []

        print("Running the calculation ...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            bands = np.load(row.path)["arr_0"]
            # save the number of bands
            n = bands.shape[0]
            # save the number of dates
            n_dates = bands.shape[2]

            # get the cloud mask out of the bands and update
            bands, cloud = self.get_clm(bands)
            # calculate the mode over each pixel and for all dates for the cloud mask
            cloud_mode = self.calculate_band_mode(cloud)
            # calculate the mean over each pixel for the band and dates
            mean = np.mean(bands, axis=0)
            # switch the bands and dates
            feature = mean.transpose(1, 0)
            # add the mode of the cloud mask back into the features
            feature = np.concatenate((feature, cloud_mode), axis=1)
            # add the features of each field to the features list
            features.append(feature)

            # get an array of the field ids, of the same size as the date array of the current feature
            field_id = np.repeat(row.field_id, feature.shape[0])
            # add the field ids array to the field ids list
            field_ids.append(field_id)
            tile_id = np.repeat(row.tile_id, feature.shape[0])
            tile_ids.append(tile_id)
            # get an array of the labels, of the same size as the date array of the current feature
            field_size = np.repeat(row.field_size, feature.shape[0])
            field_sizes.append(field_size)
            # get an array of the labels, of the same size as the date array of the current feature
            label = np.repeat(row.label, feature.shape[0])
            # add the label array to the labels list
            labels.append(label)
            # goes through the dates in each row and saves them to a list without the time [-> [:10]]
            date = [str(d)[:10] for d in row.dates]
            date = np.array(date)
            # add the date array to the dates
            dates.append(date)
            # texture metrics
            correlation = [cor for cor in row.correlation]
            list_correlation.append(correlation)
            homogeneity = [cor for cor in row.homogeneity]
            list_homogeneity.append(homogeneity)
            contrast = [cor for cor in row.contrast]
            list_contrast.append(contrast)

        # put all of the list information into an array
        all_features = np.concatenate(features)
        all_field_ids = np.concatenate(field_ids)
        all_tile_ids = np.concatenate(tile_ids)
        all_field_sizes = np.concatenate(field_sizes)
        all_dates = np.concatenate(dates)
        all_correlation = np.concatenate(list_correlation)
        all_homogeneity = np.concatenate(list_homogeneity)
        all_contrast = np.concatenate(list_contrast)
        all_labels = np.concatenate(labels)

        # put all different information into one data frame
        cols = self.get_bands()
        df_data = pd.DataFrame(all_features, columns=cols)
        df_data.insert(0, "field_id", all_field_ids)
        df_data.insert(1, "tile_id", all_tile_ids)
        df_data.insert(2, "field_size", all_field_sizes)
        df_data.insert(3, "date", all_dates)
        df_data.insert(4, "label", all_labels)
        df_data.insert(5, "correlation", all_correlation)
        df_data.insert(6, "homogeneity", all_homogeneity)
        df_data.insert(7, "contrast", all_contrast)

        # save the data frame as CSV file
        print(f"Saving the data into {self.DATA_DIR}/mean_band_perField_perDate.csv")
        df_data.to_csv(f"{self.DATA_DIR}/mean_band_perField_perDate.csv", index=False)


if __name__ == "__main__":
    from find_repo_root import get_repo_root

    ROOT_DIR = get_repo_root()
    calculation = CalculateMeanPerBand(ROOT_DIR)
    calculation.start_calculation()
