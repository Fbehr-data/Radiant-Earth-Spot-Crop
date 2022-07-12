## Modelling script to train and predict with the different models 
## Max Langer # 2022-07-11 ##

# import the needed modules
# Typer is used as CLI
import typer
import xgboost
from src.find_repo_root import get_repo_root

# set root directory
ROOT_DIR = get_repo_root()

# create a Typer object for the preprocessing
modelling = typer.Typer()

@modelling.command()
def choose_model(model_name:str):
    model_names = {"xgboost": "xgboost script"}
    print(model_names[model_name]) 


if __name__=="__main__":
    print(
        "\n \
        Commands to use: \n \
        1. train    -   Train/fit the model to the training data. \n \
        2. optimize -   Use Bayesian optimization to find the best hyperparameters. \n \
        3. predict  -   Then 'convert' the images to NPZ files for each . \n \
        4. results  -   Finally 'calculate' the mean of each band for each field for each date. \n"
        )
    modelling()