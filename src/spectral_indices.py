import numpy as np
import pandas as pd

# See Indices 
# https://www.indexdatabase.de/db/is.php?sensor_id=96

# NDVI
def cal_NDVI(Band4: pd.Series, Band8: pd.Series) -> pd.Series:
    """Takes the Bands of Sentinel2 and returns the NDVI

    Args:
        Band4 (pd.Series): Band 4 of Sentinel 2
        Band8 (pd.Series): Band 8 of Sentinel 2

    Returns:
        pd.Series: Calculated NDVI
    """
    return (Band8 - Band4) / (Band8 + Band4) 

# SIPI2
def cal_SIPI2(Band2: pd.Series, Band8: pd.Series, Band4:pd.Series) -> pd.Series:
    """Takes the Bands of Sentinel2 and returns the SIPI2

    Args:
        Band2 (pd.Series):  Band 2 of Sentinel 2
        Band8 (pd.Series):  Band 8 of Sentinel 2
        Band4 (pd.Series):  Band 4 of Sentinel 2

    Returns:
        pd.Series: Calculated SIPI
    """
    return (Band8 - Band2) / (Band8 - Band4)

# WET
def cal_WET(Band2: pd.Series, Band3: pd.Series, Band4:pd.Series, Band8: pd.Series, Band11: pd.Series, Band12:pd.Series) -> pd.Series:
    """Takes the Bands of Sentinel2 and returns the WET

    Args:
        Band2 (pd.Series): Band 2 of Sentinel 2
        Band3 (pd.Series): Band 3 of Sentinel 2
        Band4 (pd.Series): Band 4 of Sentinel 2
        Band8 (pd.Series): Band 8 of Sentinel 2
        Band11 (pd.Series): Band 11 of Sentinel 2
        Band12 (pd.Series): Band 12 of Sentinel 2

    Returns:
        pd.Series: Calculated WET
    """
    return 0.1509 * Band2 +0.1973 * Band3 + 0.3279 * Band4 + 0.3406 * Band8 - 0.7112 * Band11 - 0.4572 * Band12

# PVR
def cal_PVR(Band3: pd.Series, Band4: pd.Series) -> pd.Series:
    """Takes the Bands of Sentinel2 and returns the PVR

    Args:
        Band3 (pd.Series): Band 3 of Sentinel 2
        Band4 (pd.Series): Band 4 of Sentinel 2

    Returns:
        pd.Series:  Calculated PVR
    """
    return (Band3 - Band4) / (Band3 + Band4) 