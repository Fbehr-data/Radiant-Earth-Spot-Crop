## Preprocessing functions # Max Langer # 2022-07-06 ##

#  Import the needed modules
import numpy as np
from scipy import stats


def get_clm(bands):
  """ Extracts the cloud mask band from an array of bands.

  Args:
      bands (numpy.ndarray): Array of pixel of the current field for each band and date.

  Returns:
      bands (numpy.ndarray): Array without the cloud mask band.
      cloud (numpy.ndarray): Array of the cloud mask band.
  """
  bands_T = bands.transpose(1,0,2)                        # switch the pixel and bands
  cloud = np.expand_dims(bands_T[len(bands_T)-1],axis=0)  # get the last band, which is the cloud mask
  bands_T = bands_T[0:len(bands_T)-1]                     # remove the cloud mask band from the array
  bands = bands_T.transpose(1,0,2)                        # switch bands and pixel
  cloud = cloud.transpose(1,0,2)                          # switch bands and pixel
  return bands, cloud


def calculate_band_mode(band):
  """ Calculates the mode for a given band.

  Args:
      band (numpy.ndarray): Array of pixel of the current field for one band and each date.

  Returns:
      numpy.ndarray: Mode over all pixel for the band for each date.
  """
  mode = stats.mode(band)                             # calculate the mode
  mode = np.squeeze(mode[0],axis=0).transpose(1,0)    # reshapes the array into the form (dates, band)
  return mode