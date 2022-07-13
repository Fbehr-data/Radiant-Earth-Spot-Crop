# import the needed modules
import pandas as pd

from src.average_per_mean_function import feat_engi_date
from src.cloud_masking_function import cloud_mask
from src.spectral_indices import cal_spectral_indices, drop_na


class FeatureEngineering:
    def __init__(
        self, ROOT_DIR: str, df: pd.DataFrame, drop_unknown=False, verbose=False
    ) -> None:
        self.ROOT_DIR = ROOT_DIR
        self.df = df
        self.drop_unknown = drop_unknown
        self.verbose = verbose

    def drop_cloud_mask(self):
        print(f"Rows of Original Data: {self.df.shape[0]}")
        self.df = cloud_mask(
            self.df, drop_unknown=self.drop_unknown, verbose=self.verbose
        )
        print(f"Rows of Without Cloud Data: {self.df.shape[0]}")
        self.df.to_csv(f"{self.ROOT_DIR}/data/data_after_FE.csv", index=False)

    def calculate_spectral_indices(self):
        # Spectral Indices
        self.df = cal_spectral_indices(self.df)
        # Drop NA
        self.df = drop_na(self.df)
        self.df.to_csv(f"{self.ROOT_DIR}/data/data_after_FE.csv", index=False)

    def calculate_mean_temporal(self):
        # Mean per Month + Time-transformation
        self.df = feat_engi_date(self.df)
        # Drop April
        df_x = self.df[[s for s in self.df.columns if not "_4" in s]]
        # Drop NA
        self.df = df_x.dropna()
        # Print out Number of rows
        print(f"Lost number of fields: {df_x.shape[0] - self.df.shape[0]}")
        self.df.to_csv(f"{self.ROOT_DIR}/data/data_after_FE.csv", index=False)


if __name__ == "__main__":
    from average_per_mean_function import feat_engi_date
    from cloud_masking_function import cloud_mask
    from find_repo_root import get_repo_root
    from spectral_indices import cal_spectral_indices, drop_na

    ROOT_DIR = get_repo_root()
    df = pd.read_csv(f"{ROOT_DIR}/data/mean_band_perField_perDate.csv")

    feature_engineering = FeatureEngineering(
        ROOT_DIR=ROOT_DIR, df=df, drop_unknown=True
    )
    feature_engineering.drop_cloud_mask()
    feature_engineering.calculate_spectral_indices()
    feature_engineering.calculate_mean_temporal()
    feature_engineering.save()
