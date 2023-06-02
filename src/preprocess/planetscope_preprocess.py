import os
import shutil
from pathlib import Path

import numpy as np
from eodal.core.raster import RasterCollection

from utils import (
    MONTHS,
    DATA_DIR,
    copy_metadata,
    resize_arr,
    is_outlier
)

def remove_entries_with_no_matching_pair(dir: Path, double_entries: list) -> None:
    """
    Remove entries from the specified directory that do not have a matching pair in the double_entries list.

    Args:
        dir (Path): The directory to remove entries from.
        double_entries (list): The list of double entries to check against.

    Returns:
        None
    """
    to_remove = []
    for file in dir.iterdir():
        index = file.name[:4]
        if index.isdigit() and index not in double_entries:
            to_remove.append(file)

    for file in to_remove:
        file.unlink()

def remove_non_image_pairs(year: str, month: str) -> None:
    """
    Remove all files with no matching sentinel/planetscope file. This can occur for different reasons.

    Args:
        year (str): The year of the data.
        month (str): The month of the data.

    Returns:
        None
    """
    planetscope_dir = DATA_DIR.joinpath("processed", "planetscope", year, f"{MONTHS[month]}_{month}")
    sentinel_dir = DATA_DIR.joinpath("processed", "sentinel", year, f"{MONTHS[month]}_{month}"
                                     )
    ps_lai_dir = planetscope_dir.joinpath("lai")
    ps_metadata_dir = planetscope_dir.joinpath("metadata")
    ps_plot_dir = planetscope_dir.joinpath("plot")

    s2_lai_dir = sentinel_dir.joinpath("lai")
    s2_metadata_dir = sentinel_dir.joinpath("metadata")
    s2_plot_dir = sentinel_dir.joinpath("plot")

    if not ps_lai_dir.exists():
        raise Exception("Planetscope dir does not exsits: ", ps_lai_dir)
    if not s2_lai_dir.exists():
        raise Exception("Sentinel dir does not exist: ", s2_lai_dir)
    
    planetscope_4_band_indices = [index.name[:4] for index in ps_lai_dir.iterdir() if index.name[:4].isdigit() and index.name[5:6] == "4"]
    planetscope_8_band_indices = [index.name[:4] for index in ps_lai_dir.iterdir() if index.name[:4].isdigit() and index.name[5:6] == "8"]
    sentinel_10m_indices = [index.name[0:4] for index in s2_lai_dir.iterdir() if index.name[:4].isdigit() and index.name[5:8] == "10m"]
    sentinel_20m_indices = [index.name[0:4] for index in s2_lai_dir.iterdir() if index.name[:4].isdigit() and index.name[5:8] == "20m"]

    if planetscope_4_band_indices != planetscope_8_band_indices:
        raise Exception("Error in Planetscope Data!")
    
    if sentinel_10m_indices != sentinel_20m_indices:
        raise Exception("Error in Sentinel Data!")

    double_indices = [entry for entry in planetscope_4_band_indices if entry in sentinel_10m_indices]

    remove_entries_with_no_matching_pair(ps_lai_dir, double_indices)
    remove_entries_with_no_matching_pair(ps_metadata_dir, double_indices)
    remove_entries_with_no_matching_pair(ps_plot_dir, double_indices)
    remove_entries_with_no_matching_pair(s2_lai_dir, double_indices)
    remove_entries_with_no_matching_pair(s2_metadata_dir, double_indices)
    remove_entries_with_no_matching_pair(s2_plot_dir, double_indices)

    print(f"Number of matching files for {month} {year}: {len(double_indices)}")

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

            sizes = []
            for dir in dirs:
                curr_dir = Path(root).joinpath(dir)
                curr_files = [os.path.getsize(filename) for filename in curr_dir.iterdir() if "Analytic" in filename.name and filename.name.endswith(".tif")]
                if len(curr_files) > 1:
                    raise Exception("Error in dir ", curr_dir, "too many options to choose from")
                
                sizes.append(curr_files[0])
            max_size = max(sizes)
            max_index = sizes.index(max_size)

            if max_size < 4000000:
                print(f"Index {curr_index} is too small ({max_size}).")
                continue

            curr_dir = Path(root).joinpath(dirs[max_index])
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

def crop_data_and_save_as_np(src_data: Path, tar_dir: Path, plot_dir: Path) -> None:
    """
    Crop a PlanetScope rasterimage to a specified output dimension, normalize the pixel values, save the image as a .npy file and copy the corresponding plot

    Args:
        src_data (Path): The path to the PlanetScope image to crop and save.
        tar_dir (Path): The directory to save the .npy file to.
        plot_dir (Path): The directory to save the .png plots to.

    Returns:
        None.
    """

    index = src_data.name[:4]
    if not index.isdigit():
        raise Exception(f"Error in naming convention: {src_data}")
    
    src_plot = src_data.parent.joinpath(f"{src_data.name[:-4]}.png")
    if not src_plot.exists():
        raise Exception(f"Missing plot for {src_data.name[:4]}")
    
    out_dim = (648, 648)
    if src_data.name.endswith("4bands.tif"):
        bands = "4bands"
    elif src_data.name.endswith("8bands.tif"):
        bands = "8bands"
    else:
        raise Exception(f"Invalid file encountered: {src_data.name}")

    tar_data = tar_dir.joinpath(f"{index}_{bands}")
    tar_plot = plot_dir.joinpath(f"{index}_{bands}.png")
    raster = RasterCollection.from_multi_band_raster(src_data)
    lai = resize_arr(raster["lai"].values, out_dim)
    np.save(tar_data, lai)
    shutil.copy(src_plot, tar_plot)
                    
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
    source_dir = DATA_DIR.joinpath("filtered", "planetscope", year, f"{MONTHS[month]}_{month}", "lai")
    target_dir = DATA_DIR.joinpath("processed", "planetscope", year, f"{MONTHS[month]}_{month}", "lai")
    plot_dir = DATA_DIR.joinpath("processed", "planetscope", year, f"{MONTHS[month]}_{month}", "plot")

    target_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for file in source_dir.iterdir():
        if file.name.endswith("4bands.tif") or file.name.endswith("8bands.tif"):
            if is_outlier(file, 3):
                # print("The following file will not be used for training as it is not suited for training:", file)
                continue
            crop_data_and_save_as_np(file, target_dir, plot_dir)

    copy_metadata("planetscope", year, month)