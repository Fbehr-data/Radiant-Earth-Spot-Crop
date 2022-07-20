## Gradient Boost Model: XGBClassifier # Max Langer # 2022-07-06 ##

# import the needed modules
import os
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
# import the machine learning modules
from xgboost import XGBClassifier


class XGBModel():
    def __init__(self, ROOT_DIR:Union[str, bytes, os.PathLike], random_state=42):
        self.DATA_DIR = f"{ROOT_DIR}/data"
        self.RSEED = random_state
        np.random.seed(self.RSEED)
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame() 
        self.y_train = pd.Series(dtype="float64")
        self.y_test = pd.Series(dtype="float64")
        # initialize the GradientBoostingClassifier with optimized hyperparamters
        self.xgb = XGBClassifier(
            objective='multi:softproba',
            n_estimators=1900,
            random_state=self.RSEED,
            disable_default_eval_metric=1,
            max_depth=3,
            min_child_weight=2,
            gamma=3.535720977345191,
            colsample_bytree=0.6362412959605761,
            learning_rate=0.05986321632365238,
            reg_lambda=10.181700840893257,
            reg_alpha=8.017165700677694,
            subsample=0.3423653296694423
        )

    def load_data(self, df_train_path:Union[str, bytes, os.PathLike]="", df_test_path:Union[str, bytes, os.PathLike]=""):
        if df_train_path == "":
            self.df_train = pd.read_csv(f"{self.DATA_DIR}/Train_Dataset.csv")
        else: 
            self.df_train = pd.read_csv(df_train_path)
        if df_test_path == "":
            self.df_test = pd.read_csv(f"{self.DATA_DIR}/Test_Dataset.csv") 
        else:
            self.df_test = pd.read_csv(df_test_path)

    def train_model(self):
        # get X for the train and test data
        self.X_train = self.df_train.drop(columns=["label", "field_id", "tile_id"])
        self.X_test = self.df_test.drop(columns=["label", "field_id", "tile_id"])
        # get y for the train and test data
        y_train = self.df_train["label"]
        y_train = y_train.astype(int)
        self.y_train = y_train - 1
        y_test = self.df_test["label"]
        y_test = y_test.astype(int)
        self.y_test = y_test - 1
        print("\nStart the training...\n")
        self.xgb.fit(self.X_train, self.y_train)

    def make_prediction(self):
        print("Predicting...\n")
        # predict the absolute classes and probabilities
        y_pred_train = self.xgb.predict(self.X_train)
        y_pred_test = self.xgb.predict(self.X_test)
        # predict the probabilities for each class
        y_proba_train = self.xgb.predict_proba(self.X_train)
        y_proba_test = self.xgb.predict_proba(self.X_test)
        print("---" * 12)
        print(
            f"Accuracy on train data: {round(accuracy_score(self.y_train, y_pred_train), 3)}"
        )
        print(
            f"Accuracy on test data: {round(accuracy_score(self.y_test, y_pred_test), 3)}"
        )
        print("---" * 12)
        print(
            f'F1-score on train data: {round(f1_score(self.y_train, y_pred_train, average="macro"), 3)}'
        )
        print(
            f'F1-score on test data: {round(f1_score(self.y_test, y_pred_test, average="macro"), 3)}'
        )
        print("---" * 12)
        print(
            f"Cross-entropy on train data: {round(log_loss(self.y_train, y_proba_train), 3)}"
        )
        print(
            f"Cross-entropy on test data: {round(log_loss(self.y_test, y_proba_test), 3)}"
        )
        print("---" * 12)

if __name__ == "__main__":
    # import own modules from the scr folder
    from find_repo_root import get_repo_root
    ROOT_DIR = get_repo_root()
    xgb_model = XGBModel(ROOT_DIR)
    xgb_model.load_data()
    xgb_model.train_model()
    xgb_model.make_prediction()
