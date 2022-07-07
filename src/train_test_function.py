import numpy as np
import pandas as pd

def train_test_split_fields(df:pd.DataFrame, train_size=0.7, random_state=42) -> pd.DataFrame:
    """Splits a data frame into train and test data frames.

    Args:
        df (pd.DataFrame): The data frame to be splitted.
        train_size (float, optional): This defines the size of the train data set. Defaults to 0.7, so 70%.
        random_state (float, optional): The random seed of used for the split. 

    Returns:
        pd.DataFrame: A train and a test data frame. 
    """
    # Set a random seed
    np.random.seed(random_state)
    # Split the df into train and test using the field_ids
    n_fields = df["field_id"].nunique()
    train_fields = np.random.choice(df["field_id"].unique(), int(n_fields * train_size), replace=False)
    test_fields = df["field_id"].unique()[~np.in1d(df["field_id"].unique(), train_fields)]
    df_train = df[df["field_id"].isin(train_fields)]
    df_test = df[df["field_id"].isin(test_fields)]
    return df_train, df_test