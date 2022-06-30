import pandas as pd

def get_cloudmask_frame(df:pd.DataFrame, column:str) -> pd.DataFrame:
    """ Create a data frame which is easy to plot with the number of observations 
        for the unclouded or no information state for a given column of the data frame.

    Args:
        df (pd.DataFrame): Data frame that contains the CLM and at least another column to count the entries.
        column (str): The column to be counted.

    Returns:
        pd.DataFrame: Data frame that is ready to plot.
    """
    # create two series with the number of observations that are unclouded or have no CLM information 
    unclouded = df[df['CLM']==0].groupby(column)[column] \
    .count()
    noinfo = df[df['CLM']==255].groupby(column)[column] \
    .count()
    # create a data frame of the two series
    cloudmask_df = pd.concat([unclouded, noinfo], axis=1)
    cloudmask_df.columns = ['unclouded', 'no information']
    cloudmask_df = cloudmask_df.reset_index()
    # melts the data frame in order to make it plot-able
    cloudmask_df = cloudmask_df.melt(id_vars=column).rename(columns=str.title)
    cloudmask_df.columns = [column.title(), 'CLM', 'Count']

    return cloudmask_df


def get_info_per_field_id(df:pd.DataFrame, column:str) -> pd.DataFrame:
    """ Create a data frame with the amount of observations 
        for the given column per field_id.

    Args:
        df (pd.DataFrame):  Data frame containing the field_id and 
                            at least another column to count the observations.
        column (str):       The column to count the observations for. 

    Returns:
        pd.DataFrame: Data frame with the count per field_id.
    """
    # create a list of series with the occurrence of field_ids per month 
    counts = []
    for entry in sorted(df[column].unique()):
        count_perID = df[df[column]==entry].groupby('field_id')['field_id'].count()
        counts.append(count_perID)
    # create a data frame of list of series
    perID_df = pd.concat(counts, axis=1)
    perID_df.columns = sorted(df[column].unique())
    # melt the data frame
    perID_df = perID_df.reset_index().melt(id_vars='field_id')
    perID_df.columns = ['Field_ID', column.title(), 'Count']
    return perID_df