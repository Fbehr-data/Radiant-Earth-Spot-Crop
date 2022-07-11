## Preprocessing: 
# download, 
# file conversion to npz, 
# calculation of band means 
## Max Langer # 2022-07-11 ##

# import the needed modules
# Typer is used as CLI
import typer
from src.find_repo_root import get_repo_root
from src.preprocessing_01_download_data import PreprocessingDownload
from src.preprocessing_02_convert_to_npz import ConversionToNPZ
from src.preprocessing_03_calculate_mean_band import CalculateMeanPerBand

# set root directory
ROOT_DIR = get_repo_root()

# create a Typer object for the preprocessing
preprocessing = typer.Typer()

@preprocessing.command()
def download():
    download_process = PreprocessingDownload(ROOT_DIR)
    download_process.start_download()

@preprocessing.command()
def convert():
    conversion = ConversionToNPZ(ROOT_DIR)
    conversion.start_conversion()

@preprocessing.command()
def calculate():
    calculation = CalculateMeanPerBand(ROOT_DIR)
    calculation.start_calculation()


if __name__=="__main__":
    print(
        "\n \
        Commands to use: \n \
        1. download     -   Get started by 'download' the data. \n \
        2. convert      -   Then 'convert' the images to NPZ files for each . \n \
        3. calculate    -   Finally 'calculate' the mean of each band for each field for each date. \n"
        )
    preprocessing()