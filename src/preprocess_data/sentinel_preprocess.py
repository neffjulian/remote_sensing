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
    numpy_resize_normalize
)

np.seterr(divide='ignore', invalid='ignore')

def copy_sentinel_data(year: str, month: str) -> None:
    source_dir = DATA_DIR.joinpath("raw", "sentinel", year, f"{MONTHS[month]}_{month}")
    target_dir = DATA_DIR.joinpath("raw_filtered", "sentinel", year, f"{MONTHS[month]}_{month}")
    
    copytree(source_dir, target_dir, dirs_exist_ok=True)

def crop_data_and_save_as_np(src_10m: Path, src_20m: Path, dst_loc: Path, plot_loc: Path):
    with rasterio.open(src_10m, 'r') as data:
        b_02 = numpy_resize_normalize(data.read(1), (192, 192))
        b_03 = numpy_resize_normalize(data.read(2), (192, 192))
        b_04 = numpy_resize_normalize(data.read(3), (192, 192))

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

def check_outlier(path_to_file: Path, band: int) -> bool:
    band_size = {10: 200, 20: 100, 60: 33}
    with rasterio.open(path_to_file, 'r') as file:
        data = file.read()
        _, x, y = data.shape
        if x < band_size[band] or y < band_size[band]:
            print(path_to_file.name, x, y)
            return False

        if np.sum(data == 1) / data.size > 0.1 or np.sum(data == 0) / data.size > 0.1:
            print(path_to_file.name, np.sum(data == 1) / data.size, np.sum(data == 0) / data.size)
            return False
    return True

def save_to_np(path_10: Path, path_20: Path, path_60: Path, target_file: Path, plot_file: Path) -> None:
    with rasterio.open(path_10, 'r') as file_10m:
        blue = numpy_resize_normalize(file_10m.read(1), (192, 192))
        green = numpy_resize_normalize(file_10m.read(2), (192, 192))
        red = numpy_resize_normalize(file_10m.read(3), (192, 192))
        nir_1 = numpy_resize_normalize(file_10m.read(4), (192, 192))

    with rasterio.open(path_20, 'r') as file_20m:
        red_edge_1 = numpy_resize_normalize(file_20m.read(1), (96, 96))
        red_edge_2 = numpy_resize_normalize(file_20m.read(2), (96, 96))
        red_edge_3 = numpy_resize_normalize(file_20m.read(3), (96, 96))
        nir_2 = numpy_resize_normalize(file_20m.read(4), (96, 96))
        swir_1 = numpy_resize_normalize(file_20m.read(5), (96, 96))
        swir_2 = numpy_resize_normalize(file_20m.read(6), (96, 96))

    with rasterio.open(path_60, 'r') as file_60m:
        ultra_blue = numpy_resize_normalize(file_60m.read(1), (32, 32))
        nir_3 = numpy_resize_normalize(file_60m.read(2), (32, 32))
    np.savez(target_file,
        ultra_blue = ultra_blue,
        blue = blue,
        green = green,
        red = red,
        nir_1 = nir_1,
        nir_2 = nir_2,
        nir_3 = nir_3,
        red_edge_1 = red_edge_1,
        red_edge_2 = red_edge_2,
        red_edge_3 = red_edge_3,
        swir_1 = swir_1,
        swir_2 = swir_2
    ) 

    bgr = np.dstack((blue, green, red))
    bgr_rescaled = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(
        plot_file.as_posix(),
        bgr_rescaled
    )

def save_ndvi(path_10: Path, target_file: Path, plot_file: Path) -> None:
    with rasterio.open(path_10, 'r') as file_10m:
        red = file_10m.read(3)
        nir_1 = file_10m.read(4)

    ndvi = np.divide(red - nir_1, red + nir_1)

    ndvi_rescaled = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    np.save(
        file = target_file,
        arr = ndvi_rescaled
    )

    cv2.imwrite(
        plot_file.as_posix(),
        ndvi_rescaled
    )

def preprocess_sentinel_data(year: str, month: str) -> None:
    copy_sentinel_data(year, month)

    source_dir = DATA_DIR.joinpath("raw_filtered", "sentinel", year, f"{MONTHS[month]}_{month}", "data")
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
            
            # curr_target_plotname = plot_dir.joinpath(f"{file.name[:4]}.png")
            # crop_data_and_save_as_np(file, file_20m, curr_target_filename, curr_target_plotname)
    

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