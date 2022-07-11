## Preprocessing 03: Calculation of the mean of the bands # Max Langer # 2022-07-11 ##
## The script is based on the tutorial by the Radiant Earth Foundation for the Zindi: Spot the crop challenge.
## https://github.com/radiantearth/mlhub-tutorials/tree/main/notebooks/South%20Africa%20Crop%20Types%20Competition

# import the needed modules
import os,sys
import numpy as np 
import pandas as pd
import tarfile,json
from pathlib import Path
from tqdm.auto import tqdm
from radiant_mlhub.client import _download as download_file
from collections import OrderedDict


class PreprocessingDownload():
    """ Class for downloading the raw data 
        from the MLHUB of the Radiant Earth Foundation.
    """
    def __init__(self, ROOT_DIR:str) -> None:
        # set the directories to which the data is downloaded
        self.ROOT_DIR = ROOT_DIR
        self.DATA_DIR = f"{self.ROOT_DIR}/data"
        os.makedirs(self.DATA_DIR, exist_ok=True)
        self.IMAGE_DIR = f"{self.DATA_DIR}/images"
        os.makedirs(self.IMAGE_DIR, exist_ok=True)

        # set the important download information
        os.environ["MLHUB_API_KEY"] = "N/A"
        self.FOLDER_BASE = "ref_south_africa_crops_competition_v1"
        self.DOWNLOAD_S1 = False             # If you set this to true then the Sentinel-1 data will be downloaded
        # select which Sentinel-2 imagery bands you"d like to download here. 
        self.DOWNLOAD_S2 = OrderedDict({
            "B01": False,
            "B02": True, #Blue
            "B03": True, #Green
            "B04": True, #Red
            "B05": False,
            "B06": False,
            "B07": False,
            "B08": True, #NIR
            "B8A": False, #NIR2
            "B09": False,
            "B11": True, #SWIR1
            "B12": True, #SWIR2
            "CLM": True
        })


    def download_archive(self, archive_name:str):
        """ Downloads and extracts the raw data 
            from the MLHUB of the Radiant Earth Foundation.

        Args:
            archive_name (str): Name of the archive to be downloaded.
        """
        if os.path.exists(archive_name.replace(".tar.gz", "")):
            return
        
        print(f"Downloading {archive_name} ...")
        download_url = f"https://radiant-mlhub.s3.us-west-2.amazonaws.com/archives/{archive_name}"
        download_file(download_url, ".")
        print(f"Extracting {archive_name} ...")
        with tarfile.open(archive_name) as tfile:
            tfile.extractall()
        os.remove(archive_name)

    def resolve_path(self, base:str, path:str) -> str:
        """ Resolves the path given by the base and the specified path.
            This means that the path is converted to an absolute path.

        Args:
            base (str): Base path.
            path (str): Specified path.

        Returns:
            str: A resolved path.
        """
        return Path(os.path.join(base, path)).resolve()

    # load the data to a dataframe
    def load_df(self, collection_id:str) -> pd.DataFrame:
        """ Read the information for all images (.tif) from a .json file.

        Args:
            collection_id (str): Path to the labels folder.

        Returns:
            pd.DataFrame: Data frame containing the information for each image (.tif).
        """
        split = collection_id.split("_")[-2]
        collection = json.load(open(f"{collection_id}/collection.json", "r"))
        rows = []
        item_links = []
        for link in collection["links"]:
            if link["rel"] != "item":
                continue
            item_links.append(link["href"])
            
        for item_link in tqdm(item_links):
            item_path = f"{collection_id}/{item_link}"
            current_path = os.path.dirname(item_path)
            item = json.load(open(item_path, "r"))
            tile_id = item["id"].split("_")[-1]
            for asset_key, asset in item["assets"].items():
                rows.append([
                    tile_id,
                    None,
                    None,
                    asset_key,
                    str(self.resolve_path(current_path, asset["href"]))
                ])
                
            for link in item["links"]:
                if link["rel"] != "source":
                    continue
                source_item_id = link["href"].split("/")[-2]
                
                if source_item_id.find("_s1_") > 0 and not self.DOWNLOAD_S1:
                    continue
                elif source_item_id.find("_s1_") > 0:
                    for band in ["VV", "VH"]:
                        asset_path = Path(f"{self.FOLDER_BASE}_{split}_source_s1/{source_item_id}/{band}.tif").resolve()
                        date = "-".join(source_item_id.split("_")[10:13])
                        
                        rows.append([
                            tile_id,
                            f"{date}T00:00:00Z",
                            "s1",
                            band,
                            asset_path
                        ])
                    
                if source_item_id.find("_s2_") > 0:
                    for band, download in self.DOWNLOAD_S2.items():
                        if not download:
                            continue

                        asset_path = Path(f"{self.FOLDER_BASE}_{split}_source_s2_{band}/{source_item_id}_{band}.tif").resolve()
                        date = "-".join(source_item_id.split("_")[10:13])
                        rows.append([
                            tile_id,
                            f"{date}T00:00:00Z",
                            "s2",
                            band,
                            asset_path
                        ])

        return pd.DataFrame(rows, columns=["tile_id", "datetime", "satellite_platform", "asset", "file_path"])

    def start_download(self):
        """ Starts the downloading process. 
        """
        # change the working directory
        os.chdir(self.IMAGE_DIR)

        # start the download
        for split in ["train"]:
            # download the labels
            labels_archive = f"{self.FOLDER_BASE}_{split}_labels.tar.gz"
            self.download_archive(labels_archive)
            
            # download Sentinel-1 data
            if self.DOWNLOAD_S1:
                s1_archive = f"{self.FOLDER_BASE}_{split}_source_s1.tar.gz"
                self.download_archive(s1_archive)

            for band, download in self.DOWNLOAD_S2.items():
                if not download:
                    continue
                s2_archive = f"{self.FOLDER_BASE}_{split}_source_s2_{band}.tar.gz"
                self.download_archive(s2_archive)
        print("Finished downloading the data! \n")

        # load the info of the images into a CSV file
        print(f"Load the image info. This may take a while ... \n")
        df_images = self.load_df(f"{self.FOLDER_BASE}_train_labels")

        # save the data into a csv file
        print(f"Saving the image info into a CSV file to {self.IMAGE_DIR}/images_info_data.csv")
        df_images.to_csv("images_info_data.csv", index=False)

        # save the used bands into a pickle file
        print(f"Saving the used band info into a pkl file to {self.IMAGE_DIR}/used_bands.pkl")
        used_bands = [k for k,v in self.DOWNLOAD_S2.items() if v==True]
        used_bands = pd.DataFrame({"used_bands": used_bands})
        used_bands.to_pickle("used_bands.pkl")

        # change the working directory
        os.chdir(self.ROOT_DIR)


if __name__ == "__main__":
    from find_repo_root import get_repo_root
    ROOT_DIR = get_repo_root()
    download = PreprocessingDownload(ROOT_DIR)
    download.start_download()
    