import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from osgeo import gdal as gd

"""
Collects data from the local images/planetscope/ folder

Returns a little visualization combined with the images itself saved as a numpy array. Note the dimensions for different bands are different.
"""

########### Helper Functions ##############

def is_analytic_image(file_name):
    return "Analytic" in file_name and not file_name.endswith(".xml")

def is_metadata(file_name):
    return "metadata" in file_name

def save_image(img, location):
    img = plt.imshow(get_band_image(img, [2, 4, 6]), interpolation='lanczos')
    plt.axis('off')
    plt.savefig(location, dpi=200, bbox_inches='tight', pad_inches = 0)

def normalize(array):
    min_value, max_value = array.min(), array.max()
    return (array - min_value) / (max_value - min_value)

def brighten(band, alpha=0.13, beta=0):
    return np.clip(alpha*band+beta, 0,255)

def gammacorr(band, gamma=2):
    return np.power(band, 1/gamma)

def get_band_array(img, band):
    rasterband = img.GetRasterBand(band)
    array = rasterband.ReadAsArray()
    return normalize(brighten(gammacorr(array)))

def get_band_image(img, band_indices):
    band_arrays = [get_band_array(img, band_index) for band_index in band_indices]
    return np.dstack(band_arrays)


########### Core Functions ##############

if __name__ == '__main__':

    # Set source and target directories
    source_dir = Path('../images/planet_scope/2022')
    target_dir = Path('../preprocess_images/temp')
    
    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            curr_file = os.path.join(root, file)
            new_dir = target_dir / Path(root).name
            new_dir.mkdir(parents=True, exist_ok=True)
            
            # Process and save analytic image data and RGB plot
            if is_analytic_image(file):
                # Open the image and get image data
                image_data = gd.Open(Path(curr_file).as_posix(), gd.GA_ReadOnly)

                # Get only RGB bands and save plot (TODO: Maybe another combination might be better ?)
                image = get_band_image(image_data, [2, 4, 6])
                plt.imshow(image)
                save_image(image_data, new_dir / "plot.png")

                # Get all bands and save the data as a numpy array
                data = get_band_image(image_data, [1, 2, 3, 4, 5, 6, 7, 8])
                np.save(new_dir / "data.npy", data)

            # Copy metadata file to the target directory
            elif is_metadata(file):
                new_file = new_dir / "metadata.json"
                shutil.copyfile(curr_file, new_file)