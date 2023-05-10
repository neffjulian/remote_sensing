import os
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from eodal.core.raster import RasterCollection


from utils import (
    MONTHS,
    DATA_DIR,
    numpy_resize_normalize,
    copy_metadata
)

np.seterr(divide='ignore', invalid='ignore')

def copy_planetscope_data(year: str, month: str) -> None:
    """
    Copy PlanetScope data from the raw directory to the filtered directory,
    filtering out non-Analytic files and copying each image and its metadata file
    to the data and metadata subdirectories, respectively.

    Args:
        year (str): The year of the data to copy.
        month (str): The month of the data to copy.

    Returns:
        None.
    """
    source_dir = DATA_DIR.joinpath("raw", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("filtered", "planetscope", year, f"{MONTHS[month]}_{month}")

    if not target_dir.exists():
        data_dir = target_dir.joinpath("data")
        metadata_dir = target_dir.joinpath("metadata")
        data_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        for root, dirs, _ in os.walk(str(source_dir)):
            curr_index = Path(root).name
            if not curr_index.isdigit():
                continue

            curr_dir = Path(root).joinpath(dirs[0])
            for file in curr_dir.iterdir():
                if not "Analytic" in file.name:
                    continue

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
                
def crop_data_and_save_as_np(src: Path, target_dir: Path, plot_dir: Path) -> None:
    """
    Crop a PlanetScope image to a specified output dimension, normalize the pixel values,
    and save the image as a .npz file containing each band separately. Also compute the NDVI
    and save it as a separate .npz file. Finally, save a plot of the RGB image and a plot
    of the NDVI image as .png files.

    Args:
        src (Path): The path to the PlanetScope image to crop and save.
        target_dir (Path): The directory to save the .npz files to.
        plot_dir (Path): The directory to save the .png plots to.

    Returns:
        None.
    """
    dst_data = target_dir.joinpath(f"{src.name[:4]}.npz")
    dst_plot = plot_dir.joinpath(f"{src.name[:4]}.png")

    ndvi_data = target_dir.joinpath(f"ndvi_{src.name[:4]}.npz")
    ndvi_plot = plot_dir.joinpath(f"ndvi_{src.name[:4]}.png") 

    output_dim = (648, 648)
    raster = RasterCollection.from_multi_band_raster(src)
    b_02 = numpy_resize_normalize(raster["blue"].values, output_dim)
    b_04 = numpy_resize_normalize(raster["green"].values, output_dim)
    b_06 = numpy_resize_normalize(raster["red"].values, output_dim)
    b_07 = numpy_resize_normalize(raster["rededge"].values, output_dim)
    b_08 = numpy_resize_normalize(raster["nir"].values, output_dim)

    np.savez(dst_data, B02 = b_02, B04 = b_04, B06 = b_06, B07 = b_07, B08 = b_08 )

    bgr = np.dstack((b_02, b_04, b_06))
    bgr_rescaled = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(dst_plot.as_posix(), bgr_rescaled)

    ndvi = np.divide(b_06.astype(np.float64) - b_08.astype(np.float64), b_06.astype(np.float64) + b_08.astype(np.float64) + 0.0001)
    ndvi_rescaled = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    np.save(ndvi_data, ndvi_rescaled)
    cv2.imwrite(ndvi_plot.as_posix(), ndvi_rescaled)
                    
def preprocess_planetscope_data(year: str, month: str) -> None:
    """Preprocesses the PlanetScope data for the given year and month.

    This function reads in the raw PlanetScope data for the given year and month, crops it to a fixed size,
    and saves it as numpy arrays and plots in the processed data directory. It also copies the corresponding 
    metadata files to the processed data directory.

    Args:
        year (str): The year to preprocess the data for, in the format 'YYYY'.
        month (str): The month to preprocess the data for, in the format 'MMM'.

    Raises:
        Exception: If a file with an invalid filename is encountered in the source directory.

    Returns:
        None
    """
    source_dir = DATA_DIR.joinpath("filtered", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("processed", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    plot_dir = DATA_DIR.joinpath("processed", "planetscope", year, f"{MONTHS[month]}_{month}", "plot")

    target_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True, parents=True)

    for file in source_dir.iterdir():
        if not file.name[:4].isdigit():
            raise Exception("Invalid file found in folder", file)
        crop_data_and_save_as_np(file, target_dir, plot_dir)

    copy_metadata("planetscope", year, month)

def main(year: str, month: str) -> None:
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    
    copy_planetscope_data(year, month)
    preprocess_planetscope_data(year, month)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    main(**vars(args))