"""
All code related to preprocessing the data. This includes:
- Removing unused images
- Removing outliers
- Creating low resolution dataset
- Creating tiles from the data

@date: 2023-08-30
@author: Julian Neff, ETH Zurich

Copyright (C) 2023 Julian Neff

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import argparse
from pathlib import Path
from math import log10, sqrt

import numpy as np
import cv2
from eodal.core.raster import RasterCollection
from skimage.metrics import structural_similarity as ssim

# Set the base directory for the data
DATA_DIR = Path(__file__).parent.parent.joinpath("data")
FOLDERS = {"4b": 3, "8b": 3, "10m": 10, "20m": 20}

def get_filenames(foldername: str, in_situ: bool) -> list[str]:
    """
    Get the filenames of the files in the given folder.

    Args:
        foldername (str): The name of the folder.
        in_situ (bool): Whether the folder is in situ data or not.

    Returns:
        list[str]: The filenames of the files in the folder.
    """

    # Set the directory
    if in_situ is True:
        dir = DATA_DIR.joinpath("processed", f"{foldername}_in_situ")
    else:
        dir = DATA_DIR.joinpath("processed", foldername)

    # Return the filenames of the files in the directory
    return [file.name for file in dir.iterdir()]

def remove_files(foldername: str, files_to_keep: list[str], in_situ: bool) -> None:
    """
    Removes files with no matching counterpart in the other folder.

    Args:
        foldername (str): The name of the folder.
        files_to_keep (list[str]): The filenames of the files to keep.
        in_situ (bool): Whether the folder is in situ data or not.
    """

    # Set the directory
    if in_situ is True:
        dir = DATA_DIR.joinpath("processed", f"{foldername}_in_situ")
    else:
        dir = DATA_DIR.joinpath("processed", foldername)

    # Remove the files that are not in the list of files to keep
    files_to_remove = [file for file in dir.iterdir() if not file.name in files_to_keep]

    # Print the number of files to remove
    for file in files_to_remove:
        file.unlink()

def remove_unused_images(planetscope_bands: str, sentinel_bands: str, in_situ: bool) -> None:
    """
    Removes images that are not in both folders.

    Args:
        planetscope_bands (str): The name of the PlanetScope bands folder.
        sentinel_bands (str): The name of the Sentinel-2 bands folder.
        in_situ (bool): Whether the folder is in situ data or not.
    """

    # Get the filenames of the files in the folders
    folders = [planetscope_bands, sentinel_bands]
    files = [get_filenames(folder, in_situ) for folder in folders]
    files_to_keep = list(set.intersection(*map(set, files)))
    print(f"Equal number of files in both folders: {len(files_to_keep)}")

    # Remove the files that are not in both folders
    for folder in folders:
        remove_files(folder, files_to_keep, in_situ)

    # Check if the number of files in both folders is equal
    if in_situ is True:
        files = [[file.name for file in DATA_DIR.joinpath("processed", f"{folder}_in_situ").iterdir()] for folder in folders]
    else:
        files = [[file.name for file in DATA_DIR.joinpath("processed", folder).iterdir()] for folder in folders]

    # Check if the number of files in both folders is equal
    assert all(files[0] == files[i] for i in range(1, len(files))), "Error in original Sentinel-2 Files"

def create_tiles(data: np.ndarray, satellite: str, tiles: int = 4) -> list[np.ndarray]:
    """
    Create tiles from the given data. The number of tiles is defined by the
    number of tiles in each dimension (e.g. 4x4 tiles).

    Args:
        data (np.ndarray): The data to create tiles from.
        satellite (str): The name of the satellite. I.e. "sentinel" or "planetscope".
        tiles (int): The number of tiles in each dimension (default: 4).
    """

    # Initialize the list of tiles
    sub_data = []
    arr_x, arr_y = data.shape
    x, y = arr_x // tiles, arr_y // tiles
    
    # Set the margin for the tiles
    if satellite == "sentinel":
        margin = 10
    else: 
        margin = 60

    # Create the tiles
    for i in range(tiles):
        for j in range(tiles):

            # Set the start and end indices for the tiles
            x_start, x_end = i * x, (i + 1) * x
            y_start, y_end = j * y, (j + 1) * y

            # Add the margin to the tiles at the border
            if i == 0:
                x_start += margin
                x_end += margin
            if i == tiles - 1:
                x_start -= margin
                x_end -= margin
            if j == 0:
                y_start += margin
                y_end += margin
            if j == tiles - 1:
                y_start -= margin
                y_end -= margin

            # Add the tile to the list of tiles
            tile = data[x_start:x_end, y_start:y_end]

            # Check the number of NaN values in the tile
            if np.isnan(tile).sum() / tile.size > 0.01:
                sub_data.append(None)
            else:
                # Clip the values to the range [0, 8]
                sub_data.append(np.clip(np.nan_to_num(tile), 0., 8.))

    # Return the list of tiles
    return sub_data

def rename_in_situ_data(band: str) -> None:
    """
    Renames the in situ data files to match the field names.

    Args:
        band (str): The name of the band.

    """

    # For downloaded data to map it to the field names
    index_to_field = {'0000': 'broatefaeld', '0001': 'bramenwies', '0002': 'fluegenrain', '0003': 'hohrueti', 
                      '0004': 'altkloster', '0005': 'ruetteli', '0006': 'witzwil', '0007': 'strickhof', '0008': 'eschikon'}
    in_situ_dir = DATA_DIR.joinpath("processed", f"{band}_in_situ")

    # Iterate through the files and rename them
    for root, _, files in os.walk(in_situ_dir):
        for file in files:
            path = Path(root).joinpath(file)
            if file[3:7].isdigit():
                new_name = file.replace(file[3:7], index_to_field[file[3:7]])
                path.rename(Path(root).joinpath(new_name))

def psnr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Peak-signal-to-noise ratio (PSNR) between two images.

    Args:
        x (np.ndarray): The first image.
        y (np.ndarray): The second image.

    Returns:
        float: The PSNR between the two images.
    """

    # Return the PSNR between the two images
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))

def process_file(satellite: str, target_dir: Path, target_name: str, data: np.ndarray, in_situ: bool, augment: bool = False) -> None:
    """
    Processes a file. This includes resizing, creating tiles, filtering, and saving the tiles.
    """

    # Set the output dimensions
    if in_situ is True: # In situ data covers an area of 500x500m 
        out_dim = (25, 25) if satellite == "sentinel" else (150, 150)
    else:
        out_dim = (100, 100) if satellite == "sentinel" else (600, 600)

    # Resize the data.
    resized_data = cv2.resize(data, out_dim, interpolation=cv2.INTER_AREA)

    # Check if the data is in situ data
    if in_situ is False:

        # Create the tiles
        tiles = create_tiles(resized_data, satellite)

        # Iterate through the tiles and process them
        for i, tile in enumerate(tiles):

            # Check if the tile is None. In case of too many NaN values
            if tile is None:
                continue
            
            # Code artefact from the original code. Not used anymore.
            if augment is True:
                for j in range(4):
                    tile_rot = np.rot90(tile, j)
                    np.save(target_dir.joinpath(f"{target_name}_{i:02d}_{j:02}_00.npy"), tile_rot)
                    np.save(target_dir.joinpath(f"{target_name}_{i:02d}_{j:02}_01.npy"), np.flip(tile_rot, axis=1))
            else:
                # Save the tile
                np.save(target_dir.joinpath(f"{target_name}_{i:02d}.npy"), tile)
    else:
        # Save the data
        np.save(target_dir.joinpath(f"{target_name[3:]}.npy"), np.nan_to_num(resized_data))

def process_satellite_data(satellite: str, band: str, in_situ: bool) -> None:
    """
    Processes the satellite data. This includes resizing, creating tiles, filtering, and saving the tiles.

    Args:
        satellite (str): The name of the satellite. I.e. "sentinel" or "planetscope".
        band (str): The name of the band.
        in_situ (bool): Whether the folder is in situ data or not.
    """

    # Set the months
    months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
              "may": "05", "jun": "06", "jul": "07", "aug": "08", 
              "sep": "09", "oct": "10", "nov": "11", "dec": "12"}

    # Set the folder and minimum shape
    if in_situ is True:
        folder = DATA_DIR.joinpath("filtered", "in_situ", f"{satellite}_in_situ")
        min_shape = 25 if satellite == "sentinel" else 150
    else:
        folder = DATA_DIR.joinpath("filtered", satellite)
        min_shape = 100 if satellite == "sentinel" else 660

    # Iterate through the years and months
    for year in folder.iterdir():
        for month in year.iterdir():

            # Check if the month is in the months
            if not month.name[-3:] in months:
                continue
            
            # Set the source and target directories
            source_dir = month.joinpath("lai")

            # Check if the source directory exists
            if in_situ is True:
                print(f"Preprocess in situ data for {satellite}")
                target_dir = DATA_DIR.joinpath("processed", f"{band}_in_situ")
            else:
                print(f"Preprocess {satellite} data for {year.name} {month.name}")
                target_dir = DATA_DIR.joinpath("processed", band)

            # Create the target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)

            # Set the file ending
            file_ending = band
            file_ending += "ands.tif" if satellite == "planetscope" else "_lai.tif"

            # Iterate through the files in the source directory
            for file in source_dir.iterdir():
                if file.name.endswith(file_ending):

                    # Load the data
                    data = RasterCollection.from_multi_band_raster(file)["lai"].values

                    # Check data shape
                    if min(data.shape) < min_shape:
                        continue

                    # Set the target name
                    target_name = month.name[0:2] + "_" + file.name[0:4]

                    # Process the file
                    process_file(satellite, target_dir, target_name, data, in_situ)

def remove_outliers(ps_bands: str, s2_bands: str) -> None:
    """
    Removes outliers from the data according to the PSNR and SSIM scores.

    Args:
        ps_bands (str): The name of the PlanetScope bands folder.
        s2_bands (str): The name of the Sentinel-2 bands folder.
    """

    # Set the directories
    ps_folder = DATA_DIR.joinpath("processed", ps_bands)
    s2_folder = DATA_DIR.joinpath("processed", s2_bands)

    # Set the files to keep
    files_to_keep = []

    print(f"Number of files in {ps_folder.name}: {len(list(ps_folder.iterdir()))}")
    print(f"Number of files in {s2_folder.name}: {len(list(s2_folder.iterdir()))}")

    # Initialize the lists for the PSNR and SSIM scores
    ps_psnr_ = []
    ps_ssim_ = []

    s2_psnr_ = []
    s2_ssim_ = []

    # Iterate through the files in the PlanetScope folder
    for ps_filename in ps_folder.iterdir():

        # Load the PlanetScope file
        ps_file = np.load(ps_filename)

        # Check if the corresponding file exists in the Sentinel-2 folder
        if s2_folder.joinpath(ps_filename.name).exists():
            s2_file = np.load(s2_folder.joinpath(ps_filename.name))
        else:
            continue
        
        # Downsample the PlanetScope file
        downsampled_file = cv2.resize(ps_file, (25, 25), interpolation=cv2.INTER_AREA)

        # Re-Upsample the downsampled file
        upsampled_file = cv2.resize(downsampled_file, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Calculate the PSNR and SSIM scores between "reconstructed PlanetScope" and "original PlanetScope"
        ps_psnr = psnr(upsampled_file, ps_file)
        ps_ssim, _ = ssim((upsampled_file * (255. / 8.)).astype(np.uint8), (ps_file * (255. / 8.)).astype(np.uint8), full=True)

        # Calculate the PSNR and SSIM scores between "downsampled PlanetScope" and "original Sentinel-2"
        s2_psnr = psnr(downsampled_file, s2_file)
        s2_ssim, _ = ssim((downsampled_file * (255. / 8.)).astype(np.uint8), (s2_file * (255. / 8.)).astype(np.uint8), full=True)

        # Drop the file if the PSNR or SSIM scores are too low. Threshold was determined to only remove around 1-2% or the data.
        if ps_psnr <= 0.25 or s2_psnr <= 0.1:
            continue
        elif ps_ssim <= 0.75 or s2_ssim <= 0.1:
            continue

        # Add the file to the list of files to keep
        files_to_keep.append(ps_filename.name)

        # Add the PSNR and SSIM scores to the lists
        ps_psnr_.append(ps_psnr)
        ps_ssim_.append(ps_ssim)
        s2_psnr_.append(s2_psnr)
        s2_ssim_.append(s2_ssim)

    print(f"Number of files to keep: {len(files_to_keep)}")

    # Remove the files that are not in the list of files to keep
    remove_files(ps_folder, files_to_keep, False)
    remove_files(s2_folder, files_to_keep, False)

    # Check if the number of files in both folders is equal
    assert [file.name for file in ps_folder.iterdir()] == [file.name for file in s2_folder.iterdir()]

    print(f"Number of files in {ps_folder.name}: {len(list(ps_folder.iterdir()))}")
    print(f"Number of files in {s2_folder.name}: {len(list(s2_folder.iterdir()))}")

    # Make Plots of data

    # overlap = np.intersect1d(ps_psnr_, s2_psnr_)
    # plt.hist(s2_psnr_, bins=100, range=(0,100), label='S2', color='red', alpha=0.5)
    # plt.hist(ps_psnr_, bins=100, range=(0,100), label='PS', color='blue', alpha=0.5)
    # plt.hist(overlap, bins=100, range=(0,100), label='Overlap', color='purple', )
    # plt.legend()
    # plt.xlabel('PSNR')
    # plt.ylabel('Frequency')
    # plt.title(f'Error: Downsampled PS vs S2 and Reupsampled PS vs PS')
    # plt.show()

    # overlap = np.intersect1d(ps_ssim_, s2_ssim_)
    # plt.hist(s2_ssim_, bins=100, range=(0,1), label='S2', color='red', alpha=0.5)
    # plt.hist(ps_ssim_, bins=100, range=(0,1), label='PS', color='blue', alpha=0.5)
    # plt.hist(overlap, bins=100, range=(0,1), label='Overlap', color='purple', alpha=0.7)    
    # plt.legend()
    # plt.xlabel('SSIM')
    # plt.ylabel('Frequency')
    # plt.title(f'Error: Downsampled PS vs S2 and Reupsampled PS vs PS')
    # plt.show()

    # plt.hist(s2_psnr, bins=100, range=(0,100), label='Overlap')
    # plt.legend()
    # plt.xlabel('PSNR')
    # plt.ylabel('Frequency')
    # plt.title(f'Comparison: Downsampled PS vs. Sentinel-2')
    # plt.show()

    # plt.hist(s2_ssim, bins=100, range=(0,1), label='Overlap')
    # plt.legend()
    # plt.xlabel('SSIM')
    # plt.ylabel('Frequency')
    # plt.title(f'Comparison: Downsampled PS vs. Sentinel-2')
    # plt.show()

def create_lr_dataset(ps_band: str) -> None:
    """
    Creates a low resolution dataset by downsampling the PlanetScope data to 25x25.

    Args:
        ps_band (str): The name of the PlanetScope bands folder. I.e. "4b" or "8b".
    """

    # Set the source and target directories
    source_dir = DATA_DIR.joinpath("processed", ps_band)
    target_dir = DATA_DIR.joinpath("processed", ps_band + "_lr")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the lists for the PSNR and SSIM scores
    psnr_scores = []
    ssim_scores = []

    # Iterate through the files in the source directory
    for file in source_dir.iterdir():

        # Load the data
        data = np.load(file)

        # Downsample the data
        downsampled_data = cv2.resize(data, (25, 25), interpolation=cv2.INTER_AREA)

        # Save the data
        np.save(target_dir.joinpath(file.name), downsampled_data)

        # Re-Upsample the downsampled file
        upsampled_data = cv2.resize(downsampled_data, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Calculate the PSNR and SSIM scores between "reconstructed PlanetScope" and "original PlanetScope"
        psnr_scores.append(psnr(upsampled_data, data))
        ssim_score, _ = ssim((upsampled_data * (255. / 8.)).astype(np.uint8), (data * (255. / 8.)).astype(np.uint8), full=True)
        ssim_scores.append(ssim_score)

    print("Scores when downsampling PS to 25x25 and upsampling back to 150x150:")
    print(f"PSNR: {np.mean(psnr_scores)}")
    print(f"SSIM: {np.mean(ssim_scores)}")

def main(planetscope_bands: str, sentinel_bands: str) -> None:
    """
    Main function for preprocessing the data. Usually all 5 functions can be run directly after downloading the filtered data.

    Args:
        planetscope_bands (str): The name of the PlanetScope bands folder. I.e. "4b" or "8b".
        sentinel_bands (str): The name of the Sentinel-2 bands folder. I.e. "10m" or "20m".
    """
    # Processess the Sentinel-2 data
    process_satellite_data("sentinel", sentinel_bands, False)

    # Processess the PlanetScope data
    process_satellite_data("planetscope", planetscope_bands, False)

    # Removes images that are not available in both folders
    remove_unused_images(planetscope_bands, sentinel_bands, in_situ=False)

    # Remove outliers from the data, that is image pairs with large dissimilarities in PSNR and SSIM scores
    remove_outliers(planetscope_bands, sentinel_bands)

    # Creates a low resolution dataset by downsampling the PlanetScope data to 25x25
    create_lr_dataset(planetscope_bands)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinel_bands", type=str, required=True, help="Either '10m' or '20m'")
    parser.add_argument("--planetscope_bands", type=str, required=True, help="Either '4b' or '8b'")
    args = parser.parse_args()

    main(args.planetscope_bands, args.sentinel_bands)