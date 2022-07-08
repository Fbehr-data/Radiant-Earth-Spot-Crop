# load libraries
import pandas as pd

def drop_unknown_fun(df:pd.DataFrame, verbose:bool=False) -> pd.DataFrame:
    """Takes the Data and removes all rows with unknown Cloudinformation

    Args:
        df (pd.DataFrame): Full Dataset
        verbose (Boolean): Print information about loose of information (rows), Default to False

    Returns:
        pd.DataFrame: Dataset without clouds or dropped Cloud column
    """

    # create subset with only the data that have no cloud and drop CLM Column
    df_wo_cloud = df[df.CLM == 0]

    # Print Loose of information
    if verbose:
        print(f'Rows without unknown:             {df_wo_cloud.shape[0]}')
        print(f'Rows with unknown:                {df.shape[0]}')
        print(f'Precentage of remaining Data:     {round((df_wo_cloud.shape[0] / df.shape[0]) * 100, 3)} %')

    return df_wo_cloud

def delete_CLM_column(df:pd.DataFrame) -> pd.DataFrame:
    """Deletes the CLM column

    Args:
        df (pd.DataFrame): Data with CLM column

    Returns:
        pd.DataFrame: Data without CLM column
    """
    return df.drop('CLM', axis=1)


def cloud_mask(df:pd.DataFrame, drop_unknown:bool = False, verbose:bool = False) -> pd.DataFrame:
    """ Handle cloudy data in the dataset

    Args:
        df (pd.DataFrame): Dataset (independently from dataset)
        drop_unknown (bool): Decide whether to drop unknown data or not. Default to False
        verbose (bool, optional): Print information about loose of information (rows). Defaults to False.

    Returns:
        pd.DataFrame: returns data as df without cloudinformation
    """
    if drop_unknown:
        df = drop_unknown_fun(df, verbose)
    
    return delete_CLM_column(df)
