## Modelling script to train and predict with the different models 
## Max Langer # 2022-07-11 ##

# import the needed modules
# Typer is used as CLI
import typer
from src.find_repo_root import get_repo_root

# set root directory
ROOT_DIR = get_repo_root()

# create a Typer object for the preprocessing
modelling = typer.Typer()

def model_names():
    return ["XGBoost", "Random Forest", "K-Nearest Neighbors", "Neural Network"]

@modelling.command()
def select_model(
    model_name:str=typer.Option(
        "XGBoost", help="The model to use on the data:", autocompletion=model_names
    )
):
    typer.echo(f"You chose {model_name} as model.")



if __name__=="__main__":
    print(
        "\n \
        Commands to use: \n \
        1. download     -   Get started by 'download' the data. \n \
        2. convert      -   Then 'convert' the images to NPZ files for each . \n \
        3. calculate    -   Finally 'calculate' the mean of each band for each field for each date. \n"
        )
    modelling()