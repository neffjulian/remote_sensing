import argparse
from pathlib import Path
from shutil import copytree

import cv2
import numpy as np
from eodal.core.raster import RasterCollection

from utils import (
    MONTHS,
    DATA_DIR,
    numpy_resize_normalize
)

np.seterr(divide='ignore', invalid='ignore')

def copy_sentinel_data(year: str, month: str) -> None:
    """Copy raw Sentinel data for a given year and month to the filtered directory.

    Args:
        year (str): The year of the Sentinel data to copy.
        month (str): The month of the Sentinel data to copy, in the format 'MM'.

    Returns:
        None
    """
    source_dir = DATA_DIR.joinpath("raw", "sentinel", year, f"{MONTHS[month]}_{month}")
    target_dir = DATA_DIR.joinpath("filtered", "sentinel", year, f"{MONTHS[month]}_{month}")
    if not target_dir.exists():
        copytree(source_dir, target_dir, dirs_exist_ok=True)

def check_outlier(src: Path, band: int) -> bool:
    """
    Checks if all bands are of the correct size and that not too many entries are either black or white.

    Args:
        path_to_file (Path): The path to the raster file.
        band (int): The band to check for outliers (10, 20, or 60).

    Returns:
        bool: True if the raster file does not contain outliers in the specified band, False otherwise.
    """
    band_size = {10: 200, 20: 100, 60: 33}
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

def save_to_np(path_10: Path, path_20: Path, path_60: Path, dst: Path, plot: Path) -> None:
    """
    Save specified Sentinel raster bands as numpy arrays and a plot.

    Args:
        path_10 (Path): The path to the 10m resolution Sentinel raster file.
        path_20 (Path): The path to the 20m resolution Sentinel raster file.
        path_60 (Path): The path to the 60m resolution Sentinel raster file.
        target_file (Path): The path to save the output numpy array.
        plot_file (Path): The path to save the output plot.

    Returns:
        None.
    """
    output_dim_10 = (192, 192)
    raster_10 = RasterCollection.from_multi_band_raster(path_10)
    b_02 = numpy_resize_normalize(raster_10["B02"].values, output_dim_10)
    b_03 = numpy_resize_normalize(raster_10["B03"].values, output_dim_10)
    b_04 = numpy_resize_normalize(raster_10["B04"].values, output_dim_10)
    b_08 = numpy_resize_normalize(raster_10["B08"].values, output_dim_10)

    output_dim_20 = (96, 96)
    raster_20 = RasterCollection.from_multi_band_raster(path_20)
    b_05 = numpy_resize_normalize(raster_20["B05"].values, output_dim_20)
    b_06 = numpy_resize_normalize(raster_20["B06"].values, output_dim_20)
    b_07 = numpy_resize_normalize(raster_20["B07"].values, output_dim_20)
    b_8A = numpy_resize_normalize(raster_20["B8A"].values, output_dim_20)
    b_11 = numpy_resize_normalize(raster_20["B11"].values, output_dim_20)
    b_12 = numpy_resize_normalize(raster_20["B12"].values, output_dim_20)

    output_dim_60 = (32, 32)
    raster_60 = RasterCollection.from_multi_band_raster(path_60)    
    b_01 = numpy_resize_normalize(raster_60["B01"].values, output_dim_60)
    b_09 = numpy_resize_normalize(raster_60["B09"].values, output_dim_60)

    np.savez(dst, B01 = b_01, B02 = b_02, B03 = b_03, B04 = b_05, B06 = b_06, 
             B07 = b_07, B08 = b_08, B8A = b_8A, B09 = b_09, B11 = b_11, B12 = b_12) 

    bgr = np.dstack((b_02, b_03, b_04))
    bgr_rescaled = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(plot.as_posix(), bgr_rescaled)

def save_ndvi(path_10: Path, dst: Path, plot: Path) -> None:
    """
    Calculate and save the Normalized Difference Vegetation Index (NDVI) from a 10m resolution Sentinel raster file.

    Args:
        path_10 (Path): The path to the 10m resolution Sentinel raster file.
        target_file (Path): The path to save the output NDVI numpy array.
        plot_file (Path): The path to save the output NDVI plot.

    Returns:
        None.
    """
    raster = RasterCollection.from_multi_band_raster(path_10)
    b_04 = raster["B04"].values
    b_08 = raster["B08"].values

    ndvi = np.divide(b_04 - b_08, b_04 + b_08)
    ndvi_rescaled = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    np.save(dst, ndvi_rescaled)
    cv2.imwrite(plot.as_posix(), ndvi_rescaled)

def preprocess_sentinel_data(year: str, month: str) -> None:
    """
    Preprocess Sentinel data for a given year and month by copying the raw data, filtering and saving processed data.

    Args:
        year (str): The year of the Sentinel data to preprocess ("YYYY")
        month (str): The month of the Sentinel data to preprocess, ('MMM', e.g. 'jan' or 'feb').

    Returns:
        None.
    """
    copy_sentinel_data(year, month)

    source_dir = DATA_DIR.joinpath("filtered", "sentinel", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "data")
    plot_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "plot")

    target_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for file in source_dir.iterdir():
        if file.name.endswith("10m.tif"):
            file_20m = Path(file.as_posix().replace("10m", "20m"))
            file_60m = Path(file.as_posix().replace("10m", "60m"))

            if not (check_outlier(file, 10) and check_outlier(file_20m, 20) and check_outlier(file_60m, 60)):
                continue
            
            curr_target_filename = target_dir.joinpath(f"{file.name[:4]}.npz")
            curr_target_plotname = plot_dir.joinpath(f"{file.name[:4]}.png")
            save_to_np(file, file_20m, file_60m, curr_target_filename, curr_target_plotname)

            curr_ndvi_filename = target_dir.joinpath(f"ndvi_{file.name[:4]}.npz")
            curr_ndvi_plotname = plot_dir.joinpath(f"ndvi_{file.name[:4]}.png")
            save_ndvi(file, curr_ndvi_filename, curr_ndvi_plotname)

def main(year: str, month: str) -> None:
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    
    preprocess_sentinel_data(year, month)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    main(**vars(args))