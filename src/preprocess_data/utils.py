from pathlib import Path

import numpy as np
import cv2
import shutil

MONTHS = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
          "may": "05", "jun": "06", "jul": "07", "aug": "08", 
          "sep": "09", "oct": "10", "nov": "11", "dec": "12"}

DATA_DIR = Path().absolute().parent.parent.joinpath("data")

def crop(arr: np.ndarray, z: int) -> np.ndarray:
    x, y = arr.shape
    start_x = (x - z) // 2
    start_y = (y - z) // 2

    end_x = start_x + z
    end_y = start_y + z
    
    new_arr = np.zeros((z, z))
    new_arr[:, :] = arr[start_x:end_x, start_y:end_y]

    return new_arr

def numpy_resize_normalize(arr: np.ndarray, output_dim: tuple) -> np.ndarray:
    arr = cv2.resize(arr, output_dim, interpolation=cv2.INTER_AREA)
    arr_norm = ((arr - arr.min()) / (arr.max() - arr.min())) * 255
    return arr_norm.astype(np.uint8)

def copy_metadata(satellite: str, year: str, month: str) -> None:
    source_dir = DATA_DIR.joinpath("raw_filtered", satellite, year, f"{MONTHS[month]}_{month}", "metadata")
    target_dir = DATA_DIR.joinpath("processed", satellite, year, f"{MONTHS[month]}_{month}", "metadata")
    if not target_dir.exists():
        shutil.copytree(source_dir, target_dir)