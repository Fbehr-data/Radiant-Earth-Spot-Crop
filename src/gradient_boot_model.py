## Gradient Boost Model: XGBClassifier # Max Langer # 2022-07-06 ##

# import the needed modules
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

# import the machine learning modules
from xgboost import XGBClassifier, DMatrix, cv
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import hyperopt.pyll.stochastic
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    log_loss, 
    confusion_matrix
)

# import own modules from the scr folder
from find_repo_root import get_repo_root
from train_test_function import train_test_split_fields

# set a random seed
RSEED = 42
np.random.seed(RSEED)

# set the directories
ROOT_DIR = get_repo_root()
DATA_DIR = f"{ROOT_DIR}/data"

# load the base data from the CSV files
df = pd.read_csv(f"{DATA_DIR}/Train.csv")

# do the train-test-split
df_train, df_test = train_test_split_fields(
    df, 
    train_size=0.7, 
    random_state=RSEED
)

# get X for the train and validation data
X_train = df_train.drop(columns=["label", "field_id"])
X_val = df_test.drop(columns=["label", "field_id"])

# get y for the train and validation data
y_train = df_train["label"]
y_train = y_train.astype(int)
y_val = df_test["label"]
y_val = y_val.astype(int)

# set the class labels from 0 to 8 
y_train = y_train-1
y_val = y_val-1