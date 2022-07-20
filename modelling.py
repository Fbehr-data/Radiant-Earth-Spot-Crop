## Modelling script to train and predict with the different models 
## Max Langer # 2022-07-11 ##

# import the needed modules
# Typer is used as CLI
import typer

from src.find_repo_root import get_repo_root
from src.gradient_boot_model import XGBModel

# set root directory
ROOT_DIR = get_repo_root()

def run_xgb(ROOT_DIR):
    xgb_model = XGBModel(ROOT_DIR)
    xgb_model.load_data()
    xgb_model.train_model()
    xgb_model.make_prediction() 

# set all available models
models = {
    "xgboost": run_xgb
}

# create a Typer object for the preprocessing
modelling = typer.Typer()

@modelling.command()
def choose_model():
    print("These are the available models: \n")
    for key in models.keys():
        print(f"    {key}")
    selected_model = typer.prompt("\nWhich model to use?")
    try:
        models[selected_model](ROOT_DIR)
    except KeyError:
        print("\nThis is not a valid model choice. \nPlease choose one of the available models.")

if __name__=="__main__":
    modelling()
