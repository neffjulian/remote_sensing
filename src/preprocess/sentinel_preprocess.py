import argparse
from pathlib import Path
from shutil import copytree

import cv2
import numpy as np
from eodal.core.raster import RasterCollection

from utils import (
    MONTHS,
    DATA_DIR,
    resize_arr,
    check_outlier,
    calculate_ndvi
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


def crop_data_and_save_as_np(path_10: Path, path_20: Path, path_60: Path, data_dir: Path, plot_dir: Path) -> None:
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
    dst_data = data_dir.joinpath(f"{path_10.name[:4]}")
    dst_plot = plot_dir.joinpath(f"{path_10.name[:4]}.png")

    ndvi_data = data_dir.joinpath(f"ndvi_{path_10.name[:4]}")
    ndvi_plot = plot_dir.joinpath(f"ndvi_{path_10.name[:4]}.png") 

    output_dim_10 = (192, 192)
    raster_10 = RasterCollection.from_multi_band_raster(path_10)
    b_02 = resize_arr(raster_10["B02"].values, output_dim_10, normalize=True)
    b_03 = resize_arr(raster_10["B03"].values, output_dim_10, normalize=True)
    b_04 = resize_arr(raster_10["B04"].values, output_dim_10, normalize=True)
    b_08 = resize_arr(raster_10["B08"].values, output_dim_10, normalize=True)

    output_dim_20 = (96, 96)
    raster_20 = RasterCollection.from_multi_band_raster(path_20)
    b_05 = resize_arr(raster_20["B05"].values, output_dim_20, normalize=True)
    # b_06 = resize_arr(raster_20["B06"].values, output_dim_20, normalize=True)
    # b_07 = resize_arr(raster_20["B07"].values, output_dim_20, normalize=True)
    b_8A = resize_arr(raster_20["B8A"].values, output_dim_20, normalize=True)
    # b_11 = resize_arr(raster_20["B11"].values, output_dim_20, normalize=True)
    # b_12 = resize_arr(raster_20["B12"].values, output_dim_20, normalize=True)

    # output_dim_60 = (32, 32)
    # raster_60 = RasterCollection.from_multi_band_raster(path_60)    
    # b_01 = resize_arr(raster_60["B01"].values, output_dim_60, normalize=True)
    # b_09 = resize_arr(raster_60["B09"].values, output_dim_60, normalize=True)

    # np.savez(dst_data, B02 = b_02, B03 = b_03, B04 = b_04, 
    #          B05 = b_05, B8A = b_8A) 

    # bgr = np.dstack((b_02, b_03, b_04))
    # bgr_rescaled = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv2.imwrite(dst_plot.as_posix(), bgr_rescaled)

    ndvi = calculate_ndvi(b_04, b_08)
    ndvi_rescaled = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    np.save(ndvi_data, ndvi)
    cv2.imwrite(ndvi_plot.as_posix(), ndvi_rescaled)

def preprocess_sentinel_data(year: str, month: str) -> None:
    """
    Preprocess Sentinel data for a given year and month by copying the raw data, filtering and saving processed data.

    Args:
        year (str): The year of the Sentinel data to preprocess ("YYYY")
        month (str): The month of the Sentinel data to preprocess, ('MMM', e.g. 'jan' or 'feb').

    Returns:
        None.
    """

    source_dir = DATA_DIR.joinpath("filtered", "sentinel", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "data")
    plot_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "plot")
    
    metadata_source_dir = DATA_DIR.joinpath("filtered", "sentinel", year, f"{MONTHS[month]}_{month}", "metadata")
    metadata_target_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "metadata")
    copytree(metadata_source_dir, metadata_target_dir, dirs_exist_ok=True)

    target_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for file in source_dir.iterdir():
        if file.name.endswith("10m.tif"):
            file_20m = Path(file.as_posix().replace("10m", "20m"))
            file_60m = Path(file.as_posix().replace("10m", "60m"))

            if not (check_outlier(file, 10) and check_outlier(file_20m, 20) and check_outlier(file_60m, 60)):
                print("The following file has been removed as it is not suited for training:", file)
                continue


            crop_data_and_save_as_np(file, file_20m, file_60m, target_dir, plot_dir)

def main(year: str, month: str) -> None:
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    
    copy_sentinel_data(year, month)
    preprocess_sentinel_data(year, month)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--year", required=True, type=str)
    # parser.add_argument("--month", required=True, type=str)

    # args = parser.parse_args()
    # main(**vars(args))

    main("2022", "apr")
    main("2022", "may")
    main("2022", "jun")
    main("2022", "sep")