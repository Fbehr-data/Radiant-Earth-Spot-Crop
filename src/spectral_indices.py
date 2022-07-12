# Load libraries
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


# VARI green
def cal_VARI_green(Band2: pd.Series, Band3: pd.Series, Band4:pd.Series) -> pd.Series:
    """Takes the Bands of Sentinel2 and returns the VARI_green

    Args:
        Band2 (pd.Series): Band 2 of Sentinel 2
        Band3 (pd.Series): Band 3 of Sentinel 2
        Band4 (pd.Series): Band 4 of Sentinel 2
        

    Returns:
        pd.Series: Calculated VARI_green
    """
    return (Band3 - Band4) / (Band3 + Band4 - Band2) 


# MNSI
def cal_MNSI(Band3: pd.Series, Band4: pd.Series,Band6: pd.Series, Band9: pd.Series ) -> pd.Series:
    """Takes the Bands of Sentinel2 and returns the MNSI

    Args:
        Band3 (pd.Series): Band 3 of Sentinel 2
        Band4 (pd.Series): Band 4 of Sentinel 2
        Band6 (pd.Series): Band 6 of Sentinel 2
        Band9 (pd.Series): Band 9 of Sentinel 2
        

    Returns:
        pd.Series: Calculated MNSI
    """
    return 0.404 * Band3 - 0.039 * Band4 - 0.505 * Band6 + 0.762 * Band9 

#NDRE
def cal_NDRE (Band5: pd.Series, Band9: pd.Series) -> pd.Series:
    """Takes the Bands of Sentinel2 and returns the NDRE

    Args:
        Band5 (pd.Series): Band 5 of Sentinel 2
        Band9 (pd.Series): Band 9 of Sentinel 2
        
        

    Returns:
        pd.Series: Calculated NDRE
    """
    return (Band9 - Band5) / (Band9 + Band5) 

#GARI
def cal_GARI (Band1: pd.Series, Band3: pd.Series,Band5: pd.Series,Band9: pd.Series) -> pd.Series:
    """Takes the Bands of Sentinel2 and returns the GARI

    Args:
        Band1 (pd.Series): Band 1 of Sentinel 2
        Band3 (pd.Series): Band 3 of Sentinel 2
        Band5 (pd.Series): Band 5 of Sentinel 2
        Band9 (pd.Series): Band 9 of Sentinel 2
        
        

    Returns:
        pd.Series: Calculated GARI
    """
    return Band9 - (Band3 - (Band1 - Band5) )/ Band9 - (Band3 + (Band1 - Band5))

def cal_spectral_indices(df:pd.DataFrame) -> pd.DataFrame:
    """Takes the Data and add additional features:
        * NDVI
        * SIPI2
        * WET
        * PVR
        * VARI_green
        * MNSI
        * NDRE
        * GARI

    Args:
        df (pd.DataFrame): Full Dataset

    Returns:
        pd.DataFrame: Full Dataset with  spectral indices
    """

    # calculate Indices and PC1
    df['NDVI'] = cal_NDVI(df.B04, df.B08)
    df['SIPI2'] = cal_SIPI2(df.B02,df.B04, df.B08)
    df['WET'] = cal_WET(df.B02,df.B03, df.B04, df.B08, df.B11, df.B12)
    df['PVR'] = cal_PVR(df.B03, df.B04)
    df['VARI_green'] = cal_VARI_green(df.B02,df.B03, df.B04)
    df['MNSI'] = cal_MNSI(df.B03, df.B04,df.B06, df.B09)
    df['NDRE'] = cal_NDRE(df.B05,df.B09)
    df['GARI'] = cal_GARI(df.B01, df.B03,df.B05, df.B09)
    # Fill NA values with zero
    # df = df.fillna(value=0)
    return df 

def drop_na(df:pd.DataFrame, verbose:bool = False) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): Data with all features
        verbose (bool, optional): Print information about loose of information (rows). Defaults to False.

    Returns:
        pd.DataFrame: Data without NA
    """
    df_wo_NA = df.dropna(axis = 0)
    
    # Print Loose of information
    if verbose:
        print(f'Rows without NA:               {df_wo_NA.shape[0]}')
        print(f'Rows of Origin:                {df.shape[0]}')
        print(f'Precentage of remaining Data:  {round((df_wo_NA.shape[0] / df.shape[0]) * 100, 3)} %')

    return df_wo_NA