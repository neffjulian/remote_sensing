import argparse
from pathlib import Path
from shutil import copytree, copy

import cv2
import numpy as np
from eodal.core.raster import RasterCollection

from utils import (
    MONTHS,
    DATA_DIR,
    resize_arr,
    is_outlier
)

# np.seterr(divide='ignore', invalid='ignore')

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


def crop_data_and_save_as_np(src_data: Path, tar_dir: Path, plot_dir: Path) -> None:
    """
    Save specified Sentinel raster bands as numpy arrays and a plot.

    Args:
        data (Path): The path to the LAI resolution Sentinel raster file.
        tar_dir (Path): The path to save the output numpy array.
        plot_file (Path): The path to save the output plot.

    Returns:
        None.
    """
    index = src_data.name[:4]
    if not index.isdigit():
        raise Exception(f"Error in naming convention: {src_data}")
    
    src_plot = src_data.parent.joinpath(f"{src_data.name[:-4]}.png")
    if not src_plot.exists():
        raise Exception(f"Missing plot for {src_data.name[:4]}")
    
    if src_data.name.endswith("10m_lai.tif"):
        out_dim = (192, 192)
        res = "10m"
    elif src_data.name.endswith("20m_lai.tif"):
        out_dim = (96, 96)
        res = "20m"
    else:
        raise Exception(f"Invalid file encountered: {src_data.name}")
    
    tar_data = tar_dir.joinpath(f"{index}_{res}")
    tar_plot = plot_dir.joinpath(f"{index}_{res}")
    raster = RasterCollection.from_multi_band_raster(src_data)
    lai = resize_arr(raster["lai"].values, out_dim)
    np.save(tar_data, lai)
    copy(src_plot, tar_plot)

def preprocess_sentinel_data(year: str, month: str) -> None:
    """
    Preprocess Sentinel data for a given year and month by copying the raw data, filtering and saving processed data.

    Args:
        year (str): The year of the Sentinel data to preprocess ("YYYY")
        month (str): The month of the Sentinel data to preprocess, ('MMM', e.g. 'jan' or 'feb').

    Returns:
        None.
    """

    source_dir = DATA_DIR.joinpath("filtered", "sentinel", year, f"{MONTHS[month]}_{month}", "lai")
    target_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "lai")
    plot_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "plot")
    
    metadata_source_dir = DATA_DIR.joinpath("filtered", "sentinel", year, f"{MONTHS[month]}_{month}", "metadata")
    metadata_target_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "metadata")
    copytree(metadata_source_dir, metadata_target_dir, dirs_exist_ok=True)

    target_dir.mkdir(parents=True)
    plot_dir.mkdir(parents=True)

    for file in source_dir.iterdir():
        if file.name.endswith("10m_lai.tif"):
            if is_outlier(file, 10):
                # print("The following file will not be used for training as it is not suited for training:", file)
                continue
            crop_data_and_save_as_np(file, target_dir, plot_dir)

        if file.name.endswith("20m_lai.tif"):
            if is_outlier(file, 20):
                # print("The following file will not be used for training as it is not suited for training:", file)
                continue
            crop_data_and_save_as_np(file, target_dir, plot_dir)