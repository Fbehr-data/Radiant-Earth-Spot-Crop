print('Load modules')

import os,sys,pickle,multiprocessing,datetime
import numpy as np, pandas as pd
import tarfile,json,rasterio
from pathlib import Path
from radiant_mlhub.client import _download as download_file
from collections import OrderedDict
from tqdm.auto import tqdm
from extract_data import extract_s2_train

print('Load path')
os.environ['MLHUB_API_KEY'] = 'N/A'
OUTPUT_DIR = './data'

FOLDER_BASE = 'ref_south_africa_crops_competition_v1'
DOWNLOAD_S1 = False # If you set this to true then the Sentinel-1 data will be downloaded
# Select which Sentinel-2 imagery bands you'd like to download here. 
DOWNLOAD_S2 = OrderedDict({
    'B01': False,
    'B02': True, #Blue
    'B03': True, #Green
    'B04': True, #Red
    'B05': False,
    'B06': False,
    'B07': False,
    'B08': True, #NIR
    'B8A': False, #NIR2
    'B09': False,
    'B11': True, #SWIR1
    'B12': True, #SWIR2
    'CLM': True
})

# Set the directories
OUTPUT_DIR = f'{OUTPUT_DIR}/train'
os.makedirs(OUTPUT_DIR,exist_ok=True)
OUTPUT_DIR_BANDS = f'{OUTPUT_DIR}/bands-raw' 
os.makedirs(OUTPUT_DIR_BANDS,exist_ok=True)

df_train = pd.read_csv(f'{OUTPUT_DIR}/train_data.csv')
df_train['date'] = df_train.datetime.astype(np.datetime64)
bands = [k for k,v in DOWNLOAD_S2.items() if v==True]
print(df_train.head())

tile_ids = sorted(df_train.tile_id.unique())
print(f'extracting data from {len(tile_ids)} tiles for bands {bands}')

num_processes = multiprocessing.cpu_count()
print(f'processesing on : {num_processes} cpus')
pool = multiprocessing.Pool(num_processes)
tiles_per_process = len(tile_ids) / num_processes
tasks = []
for num_process in range(1, num_processes + 1):
    start_index = (num_process - 1) * tiles_per_process + 1
    end_index = num_process * tiles_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    sublist = tile_ids[start_index - 1:end_index]
    tasks.append((sublist,))
    print(f"Task # {num_process} process tiles {len(sublist)}")

results = []
for t in tasks:
    results.append(pool.apply_async(extract_s2_train, t))

all_results = []
for result in results:
    df = result.get()
    all_results.append(df)

df_train_meta = pd.concat(all_results)
df_train_meta['field_id'] = df_train_meta.field_id.astype(np.int32)
df_train_meta['tile_id'] = df_train_meta.field_id.astype(np.int32)
df_train_meta['label'] = df_train_meta.label.astype(np.int32)
df_train_meta = df_train_meta.sort_values(by=['field_id']).reset_index(drop=True)
df_train_meta['label'] = df_train_meta.label - 1
df_train_meta.to_pickle(f'{OUTPUT_DIR}/field_meta_train.pkl')

print(f'Training bands saved to {OUTPUT_DIR}')
print(f'Training metadata saved to {OUTPUT_DIR}/field_meta_train.pkl')