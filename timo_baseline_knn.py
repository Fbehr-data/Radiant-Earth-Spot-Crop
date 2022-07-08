'''
## African Spot Challenge based on Sentinel 2B sattelite data

The following script shows one approach to solve this challenge

using the K-neirest neighbour classifier
'''
# set the directory of the data

import sys

OUTPUT_DIR = "c:/Python"
sys.path.append("c:/Python/scr/")

# import the needed modules
import numpy as np
import pandas as pd

# import the machine learning modules
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, DMatrix, cv
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    log_loss, 
    confusion_matrix, 
    classification_report
)

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# import plotting modules and set the style
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(context="notebook", style="darkgrid", palette="crest", font="helvetica")
cmap = sns.color_palette("crest", 6) # six colors are created this way
sns.set(rc = {"figure.dpi":300})
sns.set(rc = {"figure.figsize":(6,3)})
sns.set(font_scale = 0.5)

# import own modules from the scr folder
from train_test_function import train_test_split_fields
from eda_functions import (
    plot_confusion_matrix, 
    get_label_accuracies, 
    plot_label_accuracy, 
    plot_feature_importance
)

# set a random seed
RSEED = 42
np.random.seed(RSEED)

# Set the directory of the data 
OUTPUT_DIR = 'c:/Python'
# Load the base data from the CSV files
df = pd.read_csv(f'{OUTPUT_DIR}/Train.csv')

"""
## Baseline Model

For the first base model, we only worked on the mean bands for each field and chose a K-nearest-neighbour classifier, as this is a commonly used model for raster data. 

We chose the F1 score and Accuracy as metrics, since the main goal is to correctly identify as many plants as possible. Neither FP nor FN are particularly bad or good, hence the harmonic mean F1. In addition, we also have an eye on the cross-entropy, because later we will deal with the probabilities with which a class is assigned to a field.

Here we do the train-test-split of the data.
"""

# do the validation split
df_train_val, df_test_val = train_test_split_fields(df_train_val,
                                                    train_size=0.7,
                                                    random_state=RSEED)

# get X for the train and validation data
X_train = df_train_val.drop(columns=["label", "field_id"])
X_val = df_test_val.drop(columns=["label", "field_id"])

# get y for the train and validation data
y_train = df_train_val["label"]
y_train = y_train.astype(int)
y_val = df_test_val["label"]
y_val = y_val.astype(int)

# set the classes from 0 to 8 
y_train = y_train-1
y_val = y_val-1

# Fitting the KN model
kn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
kn.fit(X_train, y_train)

# predict the absolute classes and probabilities
y_pred_train = kn.predict(X_train)
y_pred_val = kn.predict(X_val)

# predict the probabilities for each  class
y_proba_train = kn.predict_proba(X_train)
y_proba_val = kn.predict_proba(X_val)

"""Here the modelling is done."""



"""And the results of our first model. """

from sklearn.metrics import accuracy_score, f1_score, log_loss

print(f'Accuracy on train data: {accuracy_score(y_train, y_pred_train)}')
print(f'Accuracy on test data: {accuracy_score(y_train, y_pred_val)}')
print('---'*10)
print(f'F1-score on train data: {f1_score(y_train, y_pred_train, average="macro")}')
print(f'F1-score on test data: {f1_score(y_train, y_pred_val, average="macro")}')
print('---'*10)
print(f'Cross-entropy on train data: {log_loss(y_train, y_proba_train)}')
print(f'Cross-entropy on test data: {log_loss(y_train, y_proba_val)}')
print('---'*10)