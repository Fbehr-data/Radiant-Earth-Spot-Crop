## Gradient Boost Model: XGBClassifier # Max Langer # 2022-07-06 ##

# import the needed modules
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

# import the machine learning modules
from xgboost import DMatrix, XGBClassifier, cv

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
df_train, df_test = train_test_split_fields(df, train_size=0.7, random_state=RSEED)

# get X for the train and validation data
X_train = df_train.drop(columns=["label", "field_id"])
X_val = df_test.drop(columns=["label", "field_id"])

# get y for the train and validation data
y_train = df_train["label"]
y_train = y_train.astype(int)
y_val = df_test["label"]
y_val = y_val.astype(int)

# set the class labels from 0 to 8
y_train = y_train - 1
y_val = y_val - 1

# initialize the GradientBoostingClassifier
# with optimized hyperparamters
xgb = XGBClassifier(
    objective="multi:softmax",
    n_estimators=890,
    random_state=RSEED,
    disable_default_eval_metric=1,
    gpu_id=0,
    tree_method="gpu_hist",
    max_depth=10,
    min_child_weight=4,
    gamma=0.8213154931075035,
    colsample_bytree=0.6149590564567726,
    learning_rate=0.08090081872522414,
    reg_lambda=1.568502076198119,
    subsample=0.6392375791578488,
)
xgb.fit(X_train, y_train)

# predict the absolute classes and probabilities
y_pred_train = xgb.predict(X_train)
y_pred_val = xgb.predict(X_val)

# predict the probabilities for each  class
y_proba_train = xgb.predict_proba(X_train)
y_proba_val = xgb.predict_proba(X_val)

print("---" * 12)
print(f"Accuracy on train data: {round(accuracy_score(y_train, y_pred_train), 3)}")
print(f"Accuracy on test data: {round(accuracy_score(y_val, y_pred_val), 3)}")
print("---" * 12)
print(
    f'F1-score on train data: {round(f1_score(y_train, y_pred_train, average="macro"), 3)}'
)
print(
    f'F1-score on test data: {round(f1_score(y_val, y_pred_val, average="macro"), 3)}'
)
print("---" * 12)
print(f"Cross-entropy on train data: {round(log_loss(y_train, y_proba_train), 3)}")
print(f"Cross-entropy on test data: {round(log_loss(y_val, y_proba_val), 3)}")
print("---" * 12)
