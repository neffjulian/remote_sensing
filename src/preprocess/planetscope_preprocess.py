import os
import argparse
import shutil
from pathlib import Path

from skimage.exposure import match_histograms

import cv2
import numpy as np
from eodal.core.raster import RasterCollection

from utils import (
    MONTHS,
    DATA_DIR,
    copy_metadata,
    resize_arr,
    calculate_ndvi,
    check_outlier
)

def histogram_matching(year: str, month: str) -> None:
    planetscope_data = DATA_DIR.joinpath("processed", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    planetscope_plots = DATA_DIR.joinpath("processed", "planetscope", year, f"{MONTHS[month]}_{month}", "plot")
    sentinel_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}", "data")

    planetscope_indices = [index.name[:4] for index in planetscope_data.iterdir() 
                           if not index.name.startswith("ndvi") and index.name[:4].isdigit()]
    sentinel_indices = [index.name[:4] for index in sentinel_dir.iterdir() 
                        if not index.name.startswith("ndvi")]
    
    if not sentinel_indices == planetscope_indices:
        raise Exception("Error. Unequal files in Sentinel/PlanetScope dirs.")

    indices = planetscope_indices
    for index in indices:
        ps_file = planetscope_data.joinpath(f"{index}.npz")
        s2_file = sentinel_dir.joinpath(f"{index}.npz")
        ps_plot = planetscope_plots.joinpath(f"{index}.png")

        ps_data = np.load(ps_file)
        s2_data = np.load(s2_file)

        b_02 = match_histograms(ps_data["B02"], s2_data["B02"])
        b_04 = match_histograms(ps_data["B04"], s2_data["B03"])
        b_06 = match_histograms(ps_data["B06"], s2_data["B04"])
        b_07 = match_histograms(ps_data["B07"], s2_data["B05"])
        b_08 = match_histograms(ps_data["B08"], s2_data["B8A"])

        ps_bgr = np.dstack((b_02, b_04, b_06))

        cv2.imwrite(ps_plot.as_posix(), ps_bgr)
        np.savez(ps_file, B02 = b_02, B04 = b_04, B06 = b_06, B07 = b_07, B08 = b_08 )

def remove_entries_with_no_matching_pair(dir: Path, double_entries: list) -> None:
    to_remove = []
    for file in dir.iterdir():
        index = file.name[5:9]
        if index.isdigit() and index not in double_entries:
            to_remove.append(file)

    for file in to_remove:
        file.unlink()

def remove_non_image_pairs(year: str, month: str) -> None:
    planetscope_dir = DATA_DIR.joinpath("processed", "planetscope", year, f"{MONTHS[month]}_{month}")
    sentinel_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}"
                                     )
    ps_data_dir = planetscope_dir.joinpath("data")
    ps_metadata_dir = planetscope_dir.joinpath("metadata")
    ps_plot_dir = planetscope_dir.joinpath("plot")

    s2_data_dir = sentinel_dir.joinpath("data")
    s2_metadata_dir = sentinel_dir.joinpath("metadata")
    s2_plot_dir = sentinel_dir.joinpath("plot")

    if not ps_data_dir.exists():
        raise Exception("Planetscope dir does not exsits: ", ps_data_dir)
    if not s2_data_dir.exists():
        raise Exception("Sentinel dir does not exist: ", s2_data_dir)

    planetscope_indices = [index.name[5:9] for index in ps_data_dir.iterdir() 
                           if index.name[5:9].isdigit()]
    sentinel_indices = [index.name[5:9] for index in s2_data_dir.iterdir() 
                        if index.name[5:9].isdigit()]
    
    double_indices = [entry for entry in planetscope_indices if entry in sentinel_indices]
    
    remove_entries_with_no_matching_pair(ps_data_dir, double_indices)
    remove_entries_with_no_matching_pair(ps_metadata_dir, double_indices)
    remove_entries_with_no_matching_pair(ps_plot_dir, double_indices)
    remove_entries_with_no_matching_pair(s2_data_dir, double_indices)
    remove_entries_with_no_matching_pair(s2_metadata_dir, double_indices)
    remove_entries_with_no_matching_pair(s2_plot_dir, double_indices)

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
    dst_data = target_dir.joinpath(f"{src.name[:4]}")
    dst_plot = plot_dir.joinpath(f"{src.name[:4]}.png")

    ndvi_data = target_dir.joinpath(f"ndvi_{src.name[:4]}")
    ndvi_plot = plot_dir.joinpath(f"ndvi_{src.name[:4]}.png") 

    output_dim = (648, 648)
    raster = RasterCollection.from_multi_band_raster(src)
    b_02 = resize_arr(raster["blue"].values, output_dim, normalize=True)
    b_04 = resize_arr(raster["green"].values, output_dim, normalize=True)
    b_06 = resize_arr(raster["red"].values, output_dim, normalize=True)
    b_07 = resize_arr(raster["rededge"].values, output_dim, normalize=True)
    b_08 = resize_arr(raster["nir"].values, output_dim, normalize=True)

    # np.savez(dst_data, B02 = b_02, B04 = b_04, B06 = b_06, B07 = b_07, B08 = b_08 )

    # bgr = np.dstack((b_02, b_04, b_06))
    # bgr_rescaled = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv2.imwrite(dst_plot.as_posix(), bgr_rescaled)

    ndvi = calculate_ndvi(b_06, b_08)
    ndvi_rescaled = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    np.save(ndvi_data, ndvi)
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
        
        if check_outlier(file, 3):
            crop_data_and_save_as_np(file, target_dir, plot_dir)
        else:
            print("The following file has been removed as it is not suited for training:", file)

    copy_metadata("planetscope", year, month)

def main(year: str, month: str) -> None:
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    
    copy_planetscope_data(year, month)
    preprocess_planetscope_data(year, month)
    remove_non_image_pairs(year, month)
    # histogram_matching(year, month)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    main(**vars(args))

    # main("2022", "apr")
    # main("2022", "may")
    # main("2022", "jun")
    # main("2022", "sep")