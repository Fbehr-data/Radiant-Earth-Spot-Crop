# import the needed modules
from cProfile import label
from itertools import groupby
from zlib import DEF_MEM_LEVEL
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from typing import Union
import warnings
warnings.filterwarnings("ignore")

# import own modules 
if __name__ != "__main__":
    from src.train_test_function import train_test_split_fields


class ResamplingProcess():
    """ Class to resample the crop types.
    """
    def __init__(self, ROOT_DIR:str) -> None:
        # set the directory and the chunks in which the larger fields are splitted 
        self.ROOT_DIR = ROOT_DIR
        self.DATA_DIR = f"{self.ROOT_DIR}/data"

    def get_train_test_data(self
        ) -> Union[pd.DataFrame, pd.DataFrame]:
        # load the train and test data.
        df = pd.read_csv(f"{self.DATA_DIR}/data_after_FE.csv")
        df_train, df_test = train_test_split_fields(df, train_size=0.7)
        return df_train, df_test

    def merge_crop_labels(self, df_train, df_test
        ) -> Union[pd.DataFrame, pd.DataFrame]:
        # create a copy dataset with only 8 classes instead of 9 
        # 8 and 9 were combined together in one class
        df_train["label"].replace({9:8}, inplace=True)
        df_test["label"].replace({9:8}, inplace=True)
        return df_train, df_test

    def get_label_ranking(self, df_label_comb:pd.DataFrame
        ) -> pd.DataFrame:
        # rank the labels by their count
        label_ranking = df_label_comb[["label", "field_id"]] \
            .groupby("label").count().sort_values("field_id").reset_index()
        return label_ranking
    
    def get_sampling_base_label(self, label_ranking:pd.DataFrame
        ) -> int:
        # get the index and label of a label in the middle of the ranking
        sampling_base_label_idx = int((len(label_ranking)/2) - 1)
        return sampling_base_label_idx

    def get_base_label_df(self, df_label_comb:pd.DataFrame, label_ranking:pd.DataFrame
        ) -> pd.DataFrame:
        sampling_base_label_idx = self.get_sampling_base_label(label_ranking)
        sampling_base_label = label_ranking.iloc[sampling_base_label_idx]["label"]
        df_base_label = df_label_comb[df_label_comb["label"]==sampling_base_label]
        return df_base_label

    def get_resampling_size(self, label_ranking:pd.DataFrame
        ) -> int:
        # get the index of the label to resample on
        sampling_base_label_idx = self.get_sampling_base_label(label_ranking) 
        # set the sampling size to the middle label
        sampling_size = int(label_ranking.iloc[sampling_base_label_idx]["field_id"])
        return sampling_size 

    def get_classes_to_resample(self, label_ranking:pd.DataFrame
        ) -> Union[pd.Series, pd.Series]:
        sampling_base_label_idx = self.get_sampling_base_label(label_ranking) 
        classes_to_oversample = label_ranking[0:sampling_base_label_idx]["label"]
        classes_to_undersample = label_ranking[sampling_base_label_idx+1:len(label_ranking)]["label"]
        return classes_to_oversample, classes_to_undersample

    def do_oversampling(self, df_label_comb:pd.DataFrame, classes_to_oversample:pd.Series, 
        resampling_size:int) -> pd.DataFrame:
        """ Oversamples the underrepresented crop classes.
        """
        # create a subset of labels we want to oversample
        df_over = df_label_comb.loc[df_label_comb['label'].isin(classes_to_oversample)]
        # from our subset define the target and features
        X_over = df_over.drop('label',axis=1)
        y_over = df_over['label']
        # set the classes to be oversampled
        strategy = {x:resampling_size for x in classes_to_oversample}
        # start the oversampling process
        oversampler = RandomOverSampler(sampling_strategy=strategy)
        X_overS, y_overS = oversampler.fit_resample(X_over, y_over)
        print(f"oversampled labels: {Counter(y_overS)}")
        # create a frame of oversampled data by merging the above sampled feature and target
        df_overS = pd.concat([X_overS,y_overS],axis= 1)
        return df_overS

    def do_undersampling(self, df_label_comb:pd.DataFrame, classes_to_undersample:pd.Series, 
        resampling_size:int) -> pd.DataFrame:
        # create a subset of labels we want to undersample
        df_under = df_label_comb.loc[df_label_comb['label'].isin(classes_to_undersample)]
        # undersample the X_under and y_under
        X_under= df_under.drop('label',axis=1)
        y_under = df_under['label']
        # set the classes to be undersampled
        strategy = {x:resampling_size for x in classes_to_undersample}
        # start the undersampling process
        undersampler = RandomUnderSampler(sampling_strategy=strategy)
        X_underS, y_underS = undersampler.fit_resample(X_under, y_under)
        print(f"undersampled labels: {Counter(y_underS)}")
        # create a frame of undersampled data by merging the above sampled feature and target
        df_underS = pd.concat([X_underS,y_underS],axis= 1)
        return df_underS

    def save_train_test_data(self, df_oversampled:pd.DataFrame, 
        df_undersampled:pd.DataFrame, df_base_label:pd.DataFrame, df_test_label_comb:pd.DataFrame):
        # create a balanced dataset by merging the 3 subsset we created above
        df_resampled = pd.concat([df_oversampled, df_undersampled, df_base_label],axis=0)
        # get rid of inf values, which can occur due to the spectral indices calculations
        rows_to_remove_train = df_resampled.index[np.isinf(df_resampled).any(1)]
        df_resampled.drop(index=rows_to_remove_train, inplace=True)
        rows_to_remove_test = df_test_label_comb.index[np.isinf(df_test_label_comb).any(1)]
        df_test_label_comb.drop(index=rows_to_remove_test, inplace=True)
        # save the resample data set as csv file
        df_resampled.to_csv(f'{self.DATA_DIR}/Train_Dataset.csv',index=False) # create a csv file
        df_test_label_comb.to_csv(f'{self.DATA_DIR}/Test_Dataset.csv',index=False)
        print(f"Save the train and test data to {self.DATA_DIR} as Train_Dataset.csv and Test_Dataset.csv")
    
    def start_resampling(self):
        """ Run the resampling process and create a data set
            usable for training the model.
        """
        # load the train and test data
        df_train, df_test = self.get_train_test_data()
        # merge labes 8 and 9 for train and test data 
        df_train_label_comb, df_test_label_comb = self.merge_crop_labels(df_train, df_test)
        # make a ranking of the count of the labels 
        label_ranking = self.get_label_ranking(df_train_label_comb)
        # get the size to resample on
        resampling_size = self.get_resampling_size(label_ranking)
        # get a data frame for the data of the base label, on which is resampled
        df_base_label = self.get_base_label_df(df_train_label_comb, label_ranking)
        classes_to_oversample, classes_to_undersample = self.get_classes_to_resample(label_ranking)
        df_oversampled = self.do_oversampling(
            df_train_label_comb, 
            classes_to_oversample, 
            resampling_size
        )
        df_undersampled = self.do_undersampling(
            df_train_label_comb, 
            classes_to_undersample, 
            resampling_size
        )
        self.save_train_test_data(
            df_oversampled, 
            df_undersampled, 
            df_base_label,
            df_test_label_comb
        )


if __name__ == "__main__":
    from find_repo_root import get_repo_root
    from train_test_function import train_test_split_fields
    ROOT_DIR = get_repo_root()

    resampling = ResamplingProcess(ROOT_DIR=ROOT_DIR)
    resampling.start_resampling()
   

