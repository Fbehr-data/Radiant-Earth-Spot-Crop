# set the directory of the data
# depending on colab or vscode environment
on_colabs = False
import sys
if on_colabs:
  from google.colab import drive
  drive.mount("/content/drive")
  OUTPUT_DIR = "/content/drive/MyDrive/Radiant_Earth_Spot_Crop/data"
  sys.path.append("/content/drive/MyDrive/Radiant_Earth_Spot_Crop/src")
else:
  OUTPUT_DIR = "./data"
  sys.path.append("./src/")

# import the needed modules
import numpy as np
import pandas as pd

# import the machine learning modules
# The goal of this file will be the modeling based on the KNN 
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, DMatrix, cv
from sklearn.model_selection import GridSearchCV
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import hyperopt.pyll.stochastic
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

# import plotting modules and set the style for the figures
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
sns.set_theme(
    context="notebook",
    style="darkgrid",
    palette="crest",
    font="helvetica"
    )
cmap = sns.color_palette("crest", 6)    # six colors are created this way
sns.set(rc = {"figure.dpi":300})        # size of the figues and font size are created this way
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

# load the base data from the CSV files, including Train and Test dataset
dataset_name = 'Train_Dataset4'
dataset_test ='Test_Dataset.csv'
df_train = pd.read_csv(f"{OUTPUT_DIR}/{dataset_name}.csv")
df_test = pd.read_csv(f"{OUTPUT_DIR}/{dataset_test}.csv")

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

# Fitting the KN model
kn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
kn.fit(X_train, y_train)

# predict the absolute classes and probabilities
y_pred_train = kn.predict(X_train)
y_pred_val = kn.predict(X_val)

# predict the probabilities for each  class
y_proba_train = kn.predict_proba(X_train)
y_proba_val = kn.predict_proba(X_val)

# print the results for accuracy, F1-score and cross entropy 
print("---" * 12)
print(f"Accuracy on train data: {round(accuracy_score(y_train, y_pred_train), 3)}")
print(f"Accuracy on test data: {round(accuracy_score(y_val, y_pred_val), 3)}")
print("---" * 12)
print(f'F1-score on train data: {round(f1_score(y_train, y_pred_train, average="macro"), 3)}')
print(f'F1-score on test data: {round(f1_score(y_val, y_pred_val, average="macro"), 3)}')
print("---" * 12)
print(f"Cross-entropy on train data: {round(log_loss(y_train, y_proba_train), 3)}")
print(f"Cross-entropy on test data: {round(log_loss(y_val, y_proba_val), 3)}")
print("---" * 12)
