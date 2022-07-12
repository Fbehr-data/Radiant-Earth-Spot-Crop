## Preprocessing 03: Calculation of the mean of the bands # Max Langer # 2022-07-11 ##
## The script is based on the solution of Kiminya for the Zindi: Spot the crop challenge.
## https://github.com/RadiantMLHub/spot-the-crop-challenge/tree/main/2nd%20place%20-%20Kiminya

# import the needed modules
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
from collections import OrderedDict

class CalculateMeanPerBand():
    """ Class to calculate the mean per band 
        for each field for each date.
    """
    def __init__(self, ROOT_DIR:str) -> None:
        # set the directory and the chunks in which the larger fields are splitted 
        self.ROOT_DIR = ROOT_DIR
        self.DATA_DIR = f"{self.ROOT_DIR}/data"
        self.BANDS_DIR = f"{self.DATA_DIR}/bands-raw/" 
        self.IMAGE_DIR = f"{self.DATA_DIR}/images"

    def get_bands(self) -> list:
        """ Load the used bands.

        Returns:
            list: List of the used bands.
        """
        bands = pd.read_pickle(f"{self.IMAGE_DIR}/used_bands.pkl")
        bands = bands.used_bands.tolist()
        return bands

    def get_clm(self, bands:np.array):
        """ Extracts the cloud mask band from an array of bands.

        Args:
            bands (np.array): Array of pixel of the current field for each band and date.

        Returns:
            bands (np.array): Array without the cloud mask band.
            cloud (np.array): Array of the cloud mask band.
        """
        bands_T = bands.transpose(1,0,2)                        # switch the pixel and bands
        cloud = np.expand_dims(bands_T[len(bands_T)-1],axis=0)  # get the last band, which is the cloud mask
        bands_T = bands_T[0:len(bands_T)-1]                     # remove the cloud mask band from the array
        bands = bands_T.transpose(1,0,2)                        # switch bands and pixel
        cloud = cloud.transpose(1,0,2)                          # switch bands and pixel
        return bands, cloud

    def calculate_band_mode(self, band:np.array):
        """ Calculates the mode for a given band.

        Args:
            band (np.array): Array of pixel of the current field for one band and each date.

        Returns:
            np.array: Mode over all pixel for the band for each date.
        """
        mode = stats.mode(band)                             # calculate the mode
        mode = np.squeeze(mode[0],axis=0).transpose(1,0)    # reshapes the array into the form (dates, band)
        return mode

    def start_calculation(self):
        """ Start the calculation process.
        """
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

        print("Running the calculation ...")
        for _,row in tqdm(df.iterrows(), total=len(df)):
            bands = np.load(row.path)['arr_0']
            n = bands.shape[0]              # save the number of bands 
            n_dates = bands.shape[2]        # save the number of dates 

            bands, cloud = self.get_clm(bands)                       # get the cloud mask out of the bands and update
            cloud_mode = self.calculate_band_mode(cloud)             # calculate the mode over each pixel and for all dates for the cloud mask
            mean = np.mean(bands,axis=0)                        # calculate the mean over each pixel for the band and dates
            feature = mean.transpose(1,0)                       # switch the bands and dates
            feature = np.concatenate((feature, cloud_mode), axis=1)     # add the mode of the cloud mask back into the features
            features.append(feature)                            # add the features of each field to the features list
                
            field_id = np.repeat(row.field_id,feature.shape[0]) # get an array of the field ids, of the same size as the date array of the current feature
            field_ids.append(field_id)                          # add the field ids array to the field ids list
            tile_id = np.repeat(row.tile_id,feature.shape[0])
            tile_ids.append(tile_id)
            field_size = np.repeat(row.field_size,feature.shape[0])       # get an array of the labels, of the same size as the date array of the current feature
            field_sizes.append(field_size) 
            label = np.repeat(row.label,feature.shape[0])       # get an array of the labels, of the same size as the date array of the current feature
            labels.append(label)                                # add the label array to the labels list
            date = [str(d)[:10] for d in row.dates]             # goes through the dates in each row and saves them to a list without the time [-> [:10]]
            date = np.array(date)                               # convert the date list to an array
            dates.append(date)                                  # add the date array to the dates                               # add the date array to the dates

        # put all of the list information into an array
        all_features = np.concatenate(features)
        all_field_ids = np.concatenate(field_ids)
        all_tile_ids = np.concatenate(tile_ids)
        all_field_sizes = np.concatenate(field_sizes)
        all_dates = np.concatenate(dates)
        all_labels = np.concatenate(labels)

        # put all different information into one data frame
        cols = self.get_bands()
        df_data = pd.DataFrame(all_features,columns=cols)
        df_data.insert(0,"field_id",all_field_ids)
        df_data.insert(1,"tile_id",all_tile_ids)
        df_data.insert(2,"field_size", all_field_sizes)
        df_data.insert(3,"date",all_dates)
        df_data.insert(4,"label",all_labels)

        # save the data frame as CSV file
        print(f"Saving the data into {self.DATA_DIR}/mean_band_perField_perDate.csv")
        df_data.to_csv(f"{self.DATA_DIR}/mean_band_perField_perDate.csv", index=False)


if __name__ == "__main__":
    from find_repo_root import get_repo_root
    ROOT_DIR = get_repo_root()
    calculation = CalculateMeanPerBand(ROOT_DIR)
    calculation.start_calculation()
    