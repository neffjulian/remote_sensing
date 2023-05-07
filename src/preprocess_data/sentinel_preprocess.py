import os
import argparse
from pathlib import Path

import numpy as np
import rasterio
import cv2

from shutil import copytree

from utils import (
    MONTHS,
    DATA_DIR,
    numpy_resize_normalize,
    copy_metadata
)

def copy_sentinel_data(year: str, month: str) -> None:
    source_dir = DATA_DIR.joinpath("raw", "sentinel", year, f"{MONTHS[month]}_{month}")
    target_dir = DATA_DIR.joinpath("raw_filtered", "sentinel", year, f"{MONTHS[month]}_{month}")
    
    copytree(source_dir, target_dir, dirs_exist_ok=True)

def crop_data_and_save_as_np(src_10m: Path, src_20m: Path, dst_loc: Path, plot_loc: Path):
    with rasterio.open(src_10m, 'r') as data:
        b_02 = data.read(1)
        b_03 = data.read(2)
        b_04 = data.read(3)

        if min(b_02.shape, b_03.shape, b_04.shape) < (200, 200):
            raise Exception("Data with invalid shape", src_10m.name)
    
    output_dim_10m = (196, 196)
    b_02_resized = numpy_resize_normalize(b_02, output_dim_10m)
    b_03_resized = numpy_resize_normalize(b_03, output_dim_10m)
    b_04_resized = numpy_resize_normalize(b_04, output_dim_10m)
    
    with rasterio.open(src_20m, 'r') as data:
        b_05 = data.read(1)
        b_8A = data.read(2)

        if min(b_05.shape, b_8A.shape) < (100, 100):
            raise Exception("Data with invalid shape", src_20m.name)
    
    output_dim_20m = (96, 96)
    b_05_resized = numpy_resize_normalize(b_05, output_dim_20m)
    b_8A_resized = numpy_resize_normalize(b_8A, output_dim_20m)

    np.savez(dst_loc,
        Blue = b_02_resized,
        Green = b_03_resized,
        Red = b_04_resized,
        RedEdge = b_05_resized,
        NIR = b_8A_resized
    )        

    rgb = np.dstack((b_02_resized, b_03_resized, b_04_resized))
    cv2.imwrite(plot_loc.as_posix(), rgb)

def preprocess_sentinel_data(year: str, month: str) -> None:
    source_dir = DATA_DIR.joinpath("raw_filtered", "sentinel", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "data")
    plot_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "plot")

    target_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for file in source_dir.iterdir():
        if file.name.endswith("10m.tif"):
            file_20m = Path(file.as_posix().replace("10m", "20m"))
            curr_target_filename = target_dir.joinpath(f"{file.name[:4]}.npz")
            curr_target_plotname = plot_dir.joinpath(f"{file.name[:4]}.png")
            crop_data_and_save_as_np(file, file_20m, curr_target_filename, curr_target_plotname)
    
    copy_metadata("sentinel", year, month)

    

def main(year: str, month: str) -> None:
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    
    # copy_sentinel_data(year, month)
    preprocess_sentinel_data(year, month)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    main(**vars(args))