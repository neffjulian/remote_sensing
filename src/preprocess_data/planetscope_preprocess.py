import os
import argparse
from pathlib import Path

import shutil
import rasterio
import numpy as np
import cv2

from utils import (
    MONTHS,
    DATA_DIR,
    numpy_resize_normalize,
    copy_metadata
)

np.seterr(divide='ignore', invalid='ignore')


def check_sentinel_indices(year: str, month: str) -> list[str]:
    sentinel_dir = DATA_DIR.joinpath("raw", "sentinel", year, f"{MONTHS[month]}_{month}", "metadata")
    indices = [f.name[:4] for f in sentinel_dir.iterdir() if f.name[:4].isdigit()]
    print(f"Found {len(indices)} Sentinel files for {month} {year}")
    return indices

def copy_planetscope_data(year: str, month: str) -> None:
    source_dir = DATA_DIR.joinpath("raw", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("raw_filtered", "planetscope", year, f"{MONTHS[month]}_{month}")
    
    data_dir = target_dir.joinpath("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = target_dir.joinpath("metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        curr_index = Path(root).name
        if not curr_index.isdigit():
            continue

        curr_dir = Path(root).joinpath(dirs[0])

        for file in curr_dir.iterdir():
            if "Analytic" in file.name:
                if file.name.endswith(".tif"):
                    shutil.copy(src=file,
                        dst=data_dir.joinpath(f"{curr_index}.tif"),
                        follow_symlinks=False
                    )
                elif file.name.endswith(".xml"):
                    shutil.copy(
                        src=file,
                        dst=metadata_dir.joinpath(f"{curr_index}.xml"),
                        follow_symlinks=False
                    )
                else:
                    raise Exception("Error with file: ", file, " should be either a .tif file or .xml")
                
def crop_data_and_save_as_np(src: Path, dst: Path, plot_loc: Path, ndvi_file: Path, ndvi_plot: Path):
    output_dim = (648, 648)
    with rasterio.open(src, 'r') as data:
        b_02 = numpy_resize_normalize(data.read(2), output_dim)
        b_04 = numpy_resize_normalize(data.read(4), output_dim)
        b_06 = numpy_resize_normalize(data.read(6), output_dim)
        b_07 = numpy_resize_normalize(data.read(7), output_dim)
        b_08 = numpy_resize_normalize(data.read(8), output_dim)

    np.savez(dst,
        Blue = b_02,
        Green = b_04,
        Red = b_06,
        RedEdge = b_07,
        NIR = b_08
    )

    bgr = np.dstack((b_02, b_04, b_06))
    bgr_rescaled = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(plot_loc.as_posix(), bgr_rescaled)

    ndvi = np.divide(b_06.astype(np.float64) - b_08.astype(np.float64), b_06.astype(np.float64) + b_08.astype(np.float64) + 0.0001)
    ndvi_rescaled = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    np.save(
        file = ndvi_file,
        arr =  ndvi_rescaled
    )
    cv2.imwrite(
        ndvi_plot.as_posix(),
        ndvi_rescaled
    )


                    
def preprocess_planetscope_data(year, month):
    source_dir = DATA_DIR.joinpath("raw_filtered", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("processed", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    plot_dir = DATA_DIR.joinpath("processed", "planetscope", year, f"{MONTHS[month]}_{month}", "plot")

    target_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True, parents=True)

    for file in source_dir.iterdir():
        if not file.name[:4].isdigit():
            raise Exception("Invalid file found in folder", file)
        
        curr_target_filename = target_dir.joinpath(f"{file.name[:4]}.npz")
        curr_target_plotname = plot_dir.joinpath(f"{file.name[:4]}.png")

        curr_ndvi_filename = target_dir.joinpath(f"ndvi_{file.name[:4]}.npz")
        curr_ndvi_plotname = plot_dir.joinpath(f"ndvi_{file.name[:4]}.png") 

        crop_data_and_save_as_np(file, curr_target_filename, curr_target_plotname, curr_ndvi_filename, curr_ndvi_plotname)
    
    copy_metadata("planetscope", year, month)

def main(year: str, month: str) -> None:
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    
    # copy_planetscope_data(year, month)
    preprocess_planetscope_data(year, month)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    main(**vars(args))