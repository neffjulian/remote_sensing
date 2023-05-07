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

def check_sentinel_indices(year: str, month: str) -> list[str]:
    sentinel_dir = DATA_DIR.joinpath("raw", "sentinel", year, f"{MONTHS[month]}_{month}", "metadata")
    indices = [f.name[:4] for f in sentinel_dir.iterdir() if f.name[:4].isdigit()]
    print(f"Found {len(indices)} Sentinel files for {month} {year}")
    return indices

def check_planetscope_indices(source_dir):
    for file in source_dir.iterdir():
        print(file.name, type(file.name))


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
                
def crop_data_and_save_as_np(src: Path, dst: Path, plot_loc: Path):
    with rasterio.open(src, 'r') as data:
        b_02 = data.read(2)
        b_04 = data.read(4)
        b_06 = data.read(6)
        b_07 = data.read(7)
        b_08 = data.read(8)

    # output_dim = (546, 546)
    output_dim = (384, 384)
    b_02_resized = numpy_resize_normalize(b_02, output_dim)
    b_04_resized = numpy_resize_normalize(b_04, output_dim)
    b_06_resized = numpy_resize_normalize(b_06, output_dim)
    b_07_resized = numpy_resize_normalize(b_07, output_dim)
    b_08_resized = numpy_resize_normalize(b_08, output_dim)

    np.savez(dst,
        Blue = b_02_resized,
        Green = b_04_resized,
        Red = b_06_resized,
        RedEdge = b_07_resized,
        NIR = b_08_resized
    )

    rgb = np.dstack((b_02_resized, b_04_resized, b_06_resized))
    cv2.imwrite(plot_loc.as_posix(), rgb)
                    
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

        crop_data_and_save_as_np(file, curr_target_filename, curr_target_plotname)
    
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