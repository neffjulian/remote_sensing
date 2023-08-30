"""
Uses a pretrained model and band selection of Sentinel-2 data to create Super Resolved Sentinel-2 images at field boundaries.

@date: 2023-08-28
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

import argparse
import sys
from pathlib import Path

import cv2
import yaml
import numpy as np
import torch
from eodal.core.raster import RasterCollection
from eodal.core.band import Band, GeoInfo

from model.rrdb import RRDB
from model.srcnn import SRCNN
from model.edsr import EDSR
from model.esrgan import ESRGAN

sys.path.append(str(Path(__file__).parent.parent.parent.joinpath("src")))

filter_dir = Path(__file__).parent.parent.parent.joinpath('data', 'filtered')
results_dir = Path(__file__).parent.parent.parent.joinpath('data', 'results')

WEIGHT_DIR = Path(__file__).parent.parent.parent.joinpath("weights")
CONFIG_DIR = Path(__file__).parent.parent.parent.joinpath("configs")
VALIDATE_DIR = Path(__file__).parent.parent.parent.joinpath("data", "validate")

MODEL = {
    "rrdb": RRDB,
    "srcnn": SRCNN,
    "edsr": EDSR,
    "esrgan": ESRGAN
}

INDICES = ['0000', '0001', '0002', '0003', '0004', '0006', '0008', '0011', '0012', '0023', '0025', '0026', '0028', '0029', '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0040', '0046']

def get_common_indices(s2_bands, ps_bands):
    """
    Retrieve common indices (file prefixes) for which there are paired images
    for each month between Sentinel-2 and PlanetScope bands.

    Args:
        s2_bands (str): Band specification for Sentinel-2 images.
        ps_bands (str): Band specification for PlanetScope images.

    Returns:
        list: Sorted list of common indices available for each month.

    Example:
        common_indices = get_common_indices("10m", "8b")
    """

    # Define directories for Sentinel-2 and PlanetScope
    s2_dir = filter_dir.joinpath("sentinel")
    ps_dir = filter_dir.joinpath("planetscope")

    # List to store common indices for each month
    indices_per_month = []

    # Iterate through the years and months in the Sentinel-2 directory
    for year in s2_dir.iterdir():
        for month in year.iterdir():

            # Construct the corresponding directories for PlanetScope and Sentinel-2 for the current month
            curr_ps_dir = ps_dir.joinpath(year.name, month.name, "lai")
            curr_s2_dir = s2_dir.joinpath(year.name, month.name, "lai")

            # Skip to the next iteration if either directory doesn't exist
            if not curr_ps_dir.exists() or not curr_s2_dir.exists():
                continue
            
            # List LAI files in the current directories
            s2_files = list(curr_s2_dir.glob(f"*_scene_{s2_bands}_lai.tif"))
            ps_files = list(curr_ps_dir.glob(f"*_lai_{ps_bands}ands.tif"))

            # Find the common indices (file prefixes) between the two sets of files
            common_indices = list(set([f.name[0:4] for f in s2_files]).intersection([f.name[0:4] for f in ps_files]))

            # Add the common indices for this month to the list
            indices_per_month.append(common_indices)

    # Find the intersection of common indices across all months and sort them
    return sorted(list(set.intersection(*map(set, indices_per_month))), key=lambda x: int(x))

def load_model_from_checkpoint(model_name: str):
    """
    Load a pre-trained machine learning model from a checkpoint file.
    
    Args:
        model_name (str): The name of the model to load.
        
    Returns:
        torch.nn.Module: The loaded model set in evaluation mode.
    """
    
    # Open the YAML configuration file corresponding to the model
    with open(CONFIG_DIR.joinpath(f"{model_name}.yaml"), "r") as f:
        # Load hyperparameters from the YAML file
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    # Load the model from a checkpoint file
    # The architecture for the model is assumed to be defined in a dictionary named MODEL
    model = MODEL[model_name].load_from_checkpoint(
        checkpoint_path=WEIGHT_DIR.joinpath(f"{model_name}.ckpt"),
        map_location=torch.device('cpu'),  # Load the model onto CPU
        hparams=hparams  # Pass the hyperparameters from the YAML file
    )
    
    # Switch the model to evaluation mode (this is important if your model has layers like dropout, batchnorm, etc.)
    model.eval()
    
    return model

def process_files(lr: np.ndarray, model: None | SRCNN | EDSR | RRDB) -> np.ndarray:
    """
    Apply super-resolution to a low-resolution image using the specified model or bicubic interpolation.

    Args:
        lr (np.ndarray): A low-resolution image represented as a numpy array.
        model (None | SRCNN | EDSR | RRDB): A super-resolution model to apply. If None, uses bicubic interpolation.

    Returns:
        np.ndarray: A high-resolution image represented as a numpy array.
    """
    
    # Convert the low-resolution image to a PyTorch tensor and add batch and channel dimensions.
    data = torch.tensor(lr).unsqueeze(0).unsqueeze(0)
    
    # Inference step (no gradient update)
    with torch.no_grad():
        if model is None:
            # Perform bicubic interpolation if no model is specified
            out = torch.nn.functional.interpolate(data, scale_factor=6, mode="bicubic")
        else:
            # Apply the super-resolution model
            out = model(data)
    
    # Remove the batch and channel dimensions and convert back to numpy array
    return out.squeeze(0).squeeze(0).numpy()

def reconstruct_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    """
    Reconstruct a full image from a list of smaller tiles.

    Args:
        tiles (list[np.ndarray]): A list of 2D numpy arrays representing image tiles.

    Returns:
        np.ndarray: A 2D numpy array representing the reconstructed image.
    """
    
    # Get the shape of a single tile
    x, y = tiles[0].shape
    
    # Calculate the dimensions for the new reconstructed image
    new_x, new_y = x * 4, y * 4
    
    # Initialize an empty numpy array for the reconstructed image
    reconstructed = np.empty((new_x, new_y))
    
    # Loop through the reconstructed image grid and populate it with tiles
    for i in range(0, new_x, x):
        for j in range(0, new_y, y):
            # Place the next tile in the appropriate position in the reconstructed image
            reconstructed[i:i+x, j:j+y] = tiles.pop(0)
    
    return reconstructed

def create_raster(data: np.ndarray, collection: RasterCollection, out_dir: Path, create: bool = True):
    """
    Create a new raster dataset and optionally save it to disk.

    Args:
        data (np.ndarray): 2D numpy array representing the raster data.
        collection (RasterCollection): The collection object containing the geo-information.
        out_dir (Path): The directory where the raster will be saved if `create` is True.
        create (bool, optional): If True, saves the raster to disk. Defaults to True.

    Returns:
        If `create` is False, returns the created RasterCollection object.
    """

    # Create a new RasterCollection object using the supplied geo-info and data
    raster = RasterCollection(
        band_constructor=Band,
        band_name="lai",
        values=data,
        geo_info=GeoInfo(
            epsg=collection["lai"].geo_info.epsg,
            ulx=collection["lai"].geo_info.ulx,
            uly=collection["lai"].geo_info.uly,
            # New pixel resolution is 3.33 m as we have 6 times more pixels
            pixres_x=10/3,
            pixres_y=-10/3
        )
    )

    # Save the raster to disk if 'create' is True, otherwise return the raster object
    if create:
        raster.to_rasterio(out_dir)
    else:
        return raster

def process_s2_files(dir: Path, s2_files: list[Path], model: None | SRCNN | EDSR | RRDB, factor: int):
    """
    Processes a list of Sentinel-2 files and applies super-resolution on them.

    Args:
        dir (Path): The directory where the super-resolved files will be saved.
        s2_files (list[Path]): A list of Path objects representing Sentinel-2 files to process.
        model (None | SRCNN | EDSR | RRDB): The super-resolution model to use. None for bicubic interpolation.
        factor (int): The factor by which to resize the original image.
    """

    # Iterate through each Sentinel-2 file
    for path in s2_files:
        # Skip the file if it has already been processed
        if dir.joinpath(path.name[0:4] + ".tif").exists():
            continue

        # Load the multi-band raster into a RasterCollection object
        file = RasterCollection.from_multi_band_raster(path)

        # Retrieve and clean the LAI (Leaf Area Index) band
        lai = np.nan_to_num(file["lai"].values)
        shape = lai.shape

        # Downsample the image to 100x100 for processing
        ps_file_interp = cv2.resize(lai, (100, 100), interpolation=cv2.INTER_AREA)

        # Split the resized image into 25x25 tiles
        tiles = [ps_file_interp[i:i+25, j:j+25] for i in range(0, 100, 25) for j in range(0, 100, 25)]

        # Apply super-resolution on each tile and reconstruct the image
        sr_image = reconstruct_tiles([process_files(tile, model) for tile in tiles])

        # Resize the super-resolved image back to its original dimensions
        sr_file = cv2.resize(sr_image, (shape[1] * factor, shape[0] * factor), interpolation=cv2.INTER_CUBIC)

        # Create and save the super-resolved raster
        create_raster(sr_file, file, dir.joinpath(path.name[0:4] + ".tif"))

def prepare_validation_data(s2_bands: str, model_name: str):
    """
    Prepares validation data by applying super-resolution on Sentinel-2 satellite images using a specified model.

    Args:
        s2_bands (str): The Sentinel-2 band to use, e.g., "10m" or "20m".
        model_name (str): The name of the super-resolution model to use, e.g., "SRCNN", "EDSR", "RRDB". 
                          If "bicubic", bicubic interpolation will be used.
    """

    # Retrieve the indices that are common to both Sentinel-2 and PlanetScope data sets
    # Here, INDICES is hard-coded to avoid overlap with training data
    common_indices = INDICES

    # Define directories for Sentinel-2 and PlanetScope images
    s2_dir = filter_dir.joinpath("sentinel")
    ps_dir = filter_dir.joinpath("planetscope")
    
    # Output directory to save the super-resolved images
    out_dir = VALIDATE_DIR.joinpath(f"{s2_bands}_{model_name}")

    # Load the model if not using bicubic interpolation
    model = load_model_from_checkpoint(model_name) if model_name != "bicubic" else None

    # Determine the upscaling factor based on the band resolution
    upscaling_factor = 3 if s2_bands == "10m" else 6

    # Iterate through each year and month
    for year in s2_dir.iterdir():
        for month in year.iterdir():
            print(f"Processing: {month.name}")

            # Define directories for current month
            curr_ps_dir = ps_dir.joinpath(year.name, month.name, "lai")
            curr_s2_dir = s2_dir.joinpath(year.name, month.name, "lai")

            # Continue only if both directories exist
            if not curr_ps_dir.exists() or not curr_s2_dir.exists():
                continue

            # Get the Sentinel-2 files for the current month and common indices
            s2_files = [curr_s2_dir.joinpath(f"{index}_scene_{s2_bands}_lai.tif") for index in common_indices]

            # Define directory to save validation results
            validate_dir = out_dir.joinpath(month.name)
            validate_dir.mkdir(parents=True, exist_ok=True)

            # Process and super-resolve the Sentinel-2 files
            process_s2_files(validate_dir, s2_files, model, upscaling_factor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinel_bands", type=str, required=True, help="Either '10m' or '20m'")
    parser.add_argument("--model", type=str, required=True, help="Either 'bicubic', 'srcnn', 'rrdb', 'edsr' or 'esrgan'")
    args = parser.parse_args()

    prepare_validation_data(args.sentinel_bands, args.model)