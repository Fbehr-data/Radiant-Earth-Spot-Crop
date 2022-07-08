# load libraries
import pandas as pd

# Change date Format and add month and days since beginning growing season added
def calculate_mean_month_field(df:pd.DataFrame) -> pd.DataFrame:
  """Calculate the mean for each month and field_id

  Args:
      df (pd.DataFrame): Data with all features 

  Returns:
      pd.DataFrame: Data the mean for each month and field_id
  """

  # Change datatype str to Datetime of timecolumn
  df['date'] = pd.to_datetime(df['date'])
  # create relevant subset -> Calculate the month in the year
  df['month'] = df['date'].dt.month
  # Change month int to str for later column naming
  df['month'] = df['month'].apply(str)  
  
  # calculate mean for each month for each field_id
  return df.groupby(by=['field_id','month']).mean().reset_index()

def combine_feature_date(df:pd.DataFrame) -> pd.DataFrame:
  """ Transform the date (month) to each feature, so feature and time is combined

  Args:
      df (pd.DataFrame): data with mean per month and per field

  Returns:
      pd.DataFrame: returned transformed data
  """
  # get list of features
  features = list(set(df) - set(['field_id', 'month', 'label']))

  # pivot for each month over field id --> Combine feature and Time
  df_res = df.pivot(index='field_id', columns='month', values=features).reset_index()

  # change column names
  df_res.columns = ['_'.join(col).strip() for col in df_res.columns.values]

  # Left join with labels 
  merge = df_res.merge(df[['field_id', 'label']].applymap(int), left_on='field_id_', right_on='field_id', how ='left')

  # drop field_id_ column
  return merge.drop('field_id_', axis =1).drop_duplicates()

def feat_engi_date(df:pd.DataFrame) -> pd.DataFrame:
  """includes two main function --> Full Feature Engineering
      * Calculate the mean for each month and field_id
      * Transform the date (month) to each feature, so feature and time is combined

  Args:
      df (pd.DataFrame): initial dataset

  Returns:
      pd.DataFrame: transformed dataset
  """
  return combine_feature_date(calculate_mean_month_field(df))
