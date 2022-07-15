# Import the needed modules
import numpy as np
import pandas as pd
from scipy import stats 
from scipy.ndimage import gaussian_filter
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import hyperopt.pyll.stochastic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    log_loss, 
    confusion_matrix, 
    classification_report
)
from find_repo_root import get_repo_root
 # Set the directory of the data 
ROOT_DIR = get_repo_root()
OUTPUT_DIR = f"{ROOT_DIR}/data"
# Load the base data from the CSV files
df_train = pd.read_csv(f'{OUTPUT_DIR}/Train_Dataset.csv')
df_test = pd.read_csv(f'{OUTPUT_DIR}/Test_Dataset.csv')

# Set a random seed
RSEED = 42
np.random.seed(RSEED)

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
labels = y_train.unique()
# Fitting the RF model
rf = RandomForestClassifier(n_estimators = 200, random_state = RSEED, n_jobs = -1, verbose=1,max_features='auto', max_depth=10, criterion='entropy')
rf.fit(X_train, y_train)

# predict the absolute classes and probabilities
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_val)
y_proba_train = rf.predict_proba(X_train)
y_proba_test = rf.predict_proba(X_val)


print(f'Accuracy on train data: {accuracy_score(y_train, y_pred_train)}')
print(f'Accuracy on test data: {accuracy_score(y_val, y_pred_test)}')
print('---'*10)
print(f'F1-score on train data: {f1_score(y_train, y_pred_train, average="macro")}')
print(f'F1-score on test data: {f1_score(y_val, y_pred_test, average="macro")}')
print('---'*10)
print(f'Cross-entropy on train data: {log_loss(y_train, y_proba_train, labels=labels)}')
print(f'Cross-entropy on test data: {log_loss(y_val, y_proba_test, labels=labels)}')
print('---'*10)