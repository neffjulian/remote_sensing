import os
from pathlib import Path

import numpy as np

DATA_DIR = Path().absolute().parent.parent.joinpath('data', 'processed')

MONTHS = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
          "may": "05", "jun": "06", "jul": "07", "aug": "08", 
          "sep": "09", "oct": "10", "nov": "11", "dec": "12"}

def check_available_pairs(year: str, month: str) -> list:
    planetscope_dir = DATA_DIR.joinpath("planetscope", year, f"{MONTHS[month]}_{month}", "data")
    sentinel_dir = DATA_DIR.joinpath("sentinel", year, f"{MONTHS[month]}_{month}", "data")

    if not planetscope_dir.exists() or not sentinel_dir.exists():
        raise Exception("Data not available for given month year combination.")
    
    planetscope_indices = [index.name[:4] for index in planetscope_dir.iterdir() 
                           if not index.name.startswith("ndvi") and index.name[:4].isdigit()]
    sentinel_indices = [index.name[:4] for index in sentinel_dir.iterdir() 
                        if not index.name.startswith("ndvi")]

    indices = sorted(list(set(planetscope_indices + sentinel_indices)))

    get_data_pair(year, month, "0000")

    return indices

def cut_np_array(data: np.ndarray) -> list:
    x, y = data.shape
    new_x, new_y = int(x/4), int(y/4)
    
    sub_data = []
    for i in range(4):
        for j in range(4):
            sub_data.append(data[(i * new_x):((i+1) * new_x), (j * new_y):((j+1) * new_y)])
    return sub_data
        

def get_data_pair(year: str, month: str, index: str) -> list:
    planetscope_data = DATA_DIR.joinpath("planetscope", year, f"{MONTHS[month]}_{month}", "data", f"{index}.npz")
    sentinel_data = DATA_DIR.joinpath("sentinel", year, f"{MONTHS[month]}_{month}", "data", f"{index}.npz")

    ps_data = np.load(planetscope_data)
    s2_data = np.load(sentinel_data)
    ps_b02 = ps_data["B02"]
    s2_b02 = s2_data["blue"]
    cut_np_array(ps_b02)
