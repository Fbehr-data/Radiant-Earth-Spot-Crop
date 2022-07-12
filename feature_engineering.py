## Feature engineering: 
# work with the cloud mask, 
# calculation of indices, 
# calculation of the mean per month for each band and index
## Felix Behrendt & Max Langer # 2022-07-11 ##

# import the needed modules
# Typer is used as CLI
import typer
import pandas as pd
from src.find_repo_root import get_repo_root
from src.feature_engineering_class import FeatureEngineering
from src.resampling_crop_type import ResamplingProcess

# set root directory
ROOT_DIR = get_repo_root()

# create a typer object for the preprocessing
engineering = typer.Typer()

@engineering.command()
def dropcloud():
    df = pd.read_csv(f"{ROOT_DIR}/data/mean_band_perField_perDate.csv")
    feature_engineering = FeatureEngineering(ROOT_DIR=ROOT_DIR, df=df, drop_unknown=True)
    feature_engineering.drop_cloud_mask()

@engineering.command()
def calindices():
    df = pd.read_csv(f"{ROOT_DIR}/data/data_after_FE.csv")
    feature_engineering = FeatureEngineering(ROOT_DIR=ROOT_DIR, df=df, drop_unknown=True)
    feature_engineering.calculate_spectral_indices()

@engineering.command()
def calmean():
    df = pd.read_csv(f"{ROOT_DIR}/data/data_after_FE.csv")
    feature_engineering = FeatureEngineering(ROOT_DIR=ROOT_DIR, df=df, drop_unknown=True)
    feature_engineering.calculate_mean_temporal()

@engineering.command()
def resampling():
    resampling = ResamplingProcess(ROOT_DIR=ROOT_DIR)
    resampling.start_resampling()


if __name__=="__main__":
    print(
        "\n \
        Commands to use: \n \
        1. dropcloud    -   Get started by 'download' the data. \n \
        2. calindices   -   Then 'convert' the images to NPZ files for each . \n \
        3. calmean      -   Finally 'calculate' the mean of each band for each field for each date. \n \
        4. resampling   -   Due to the data imbalance a 'resampling' is needed."
        )
    engineering()
