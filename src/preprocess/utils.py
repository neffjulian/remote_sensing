from pathlib import Path

import numpy as np
import cv2
import shutil
from eodal.core.raster import RasterCollection

np.seterr(divide='ignore', invalid='ignore')

MONTHS = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
          "may": "05", "jun": "06", "jul": "07", "aug": "08", 
          "sep": "09", "oct": "10", "nov": "11", "dec": "12"}

DATA_DIR = Path().absolute().parent.parent.joinpath("data")

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)

    ndvi = (nir - red) / (nir + red)
    return ndvi

def crop(arr: np.ndarray, z: int) -> np.ndarray:
    """Crops a 2D numpy array to a square of size `z` centered on the original array.

    Args:
        arr (np.ndarray): The input array to be cropped.
        z (int): The size of the square to crop the input array to.

    Returns:
        np.ndarray: The cropped array as a new 2D numpy array.
    """
    x, y = arr.shape
    start_x = (x - z) // 2
    start_y = (y - z) // 2

    end_x = start_x + z
    end_y = start_y + z
    
    new_arr = np.zeros((z, z))
    new_arr[:, :] = arr[start_x:end_x, start_y:end_y]

    return new_arr

def resize_arr(arr: np.ndarray, output_dim: tuple, normalize: bool = False) -> np.ndarray:
    """Resizes a 2D numpy array to the specified output dimension.

    Args:
        arr (np.ndarray): The input array to be resized and normalized.
        output_dim (tuple): A tuple of integers (height, width) specifying the output dimension.
        normalize (bool): If the values should also be resized
    Returns:
        np.ndarray: The resized and normalized array as a new 2D numpy array with dtype np.uint8.
    """
    out = cv2.resize(arr, output_dim, interpolation=cv2.INTER_AREA)
    if normalize is True:
        out = ((out - out.min()) / (out.max() - out.min())) * 255
    return out.astype(np.uint8)

def check_outlier(src: Path, band: int) -> bool:
    """
    Checks if all bands are of the correct size and that not too many entries are either black or white.

    Args:
        path_to_file (Path): The path to the raster file.
        band (int): The band to check for outliers (10, 20, or 60).

    Returns:
        bool: True if the raster file does not contain outliers in the specified band, False otherwise.
    """
    band_size = {3: 660, 10: 200, 20: 100, 60: 33}
    raster = RasterCollection.from_multi_band_raster(src)

    for band_name in raster.band_names:
        data = raster[band_name].values
        x, y = data.shape

        if min(x, y) < band_size[band]:
            return False
        
        percentage_ones =  (np.sum(data == 1) / data.size) * 100
        percentage_zeros = (np.sum(data == 0) / data.size) * 100

        if max(percentage_ones, percentage_zeros) > 10:
            return False

    return True

def copy_metadata(satellite: str, year: str, month: str) -> None:
    """Copy the metadata folder from the raw data directory to the processed data directory.

    Args:
        satellite (str): The name of the satellite, either "planetscope" or "sentinel".
        year (str): The year of the data to be processed.
        month (str): The month of the data to be processed.

    Returns:
        None
    """
    source_dir = DATA_DIR.joinpath("filtered", satellite, year, f"{MONTHS[month]}_{month}", "metadata")
    target_dir = DATA_DIR.joinpath("processed", satellite, year, f"{MONTHS[month]}_{month}", "metadata")
    if not target_dir.exists():
        shutil.copytree(source_dir, target_dir)