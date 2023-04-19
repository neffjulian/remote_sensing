import os
from pathlib import Path
import osgeo.gdal as gd
import numpy as np
import matplotlib.pyplot as plt

def is_analytic_image(file_name: str) -> bool:
    return "Analytic" in file_name and file_name.endswith(".tif")

def save_image(img, location: Path) -> None:
    img = plt.imshow(get_band_image(img, [2, 4, 6]), interpolation='lanczos')
    plt.axis('off')
    plt.savefig(location, dpi=200, bbox_inches='tight', pad_inches = 0)

def normalize(array: np.ndarray) -> np.ndarray:
    min_value, max_value = array.min(), array.max()
    return (array - min_value) / (max_value - min_value)

def brighten(band: np.ndarray, alpha: float = 0.13, beta: float = 0) -> np.ndarray:
    return np.clip(alpha*band+beta, 0,255)

def gammacorr(band: np.ndarray, gamma=2) -> np.ndarray:
    return np.power(band, 1/gamma)

def get_band_array(img, band):
    rasterband = img.GetRasterBand(band)
    array = rasterband.ReadAsArray()
    return normalize(brighten(gammacorr(array)))

def get_band_image(img, band_indices):
    band_arrays = [get_band_array(img, band_index) for band_index in band_indices]
    return np.dstack(band_arrays)

def crop(arr: np.ndarray, z: int) -> np.ndarray:
    _, x, y = arr.shape
    start_x = (x - z) // 2
    start_y = (y - z) // 2
    
    end_x = start_x + z
    end_y = start_y + z
    
    new_arr = np.zeros((1, z, z))
    new_arr[0,:,:] = arr[0,start_x:end_x,start_y:end_y]
    return new_arr

def get_data_and_plot(month: str, year: str):
    source_dir = Path(f'images/planet_scope/{year}_{month}')

    for root, _, files in os.walk(source_dir):
        # TODO: Once only one image is downloaded per scene change target directory here
        target_dir = Path(root).parent

        for file in files:
            curr_file = Path(root, file)
            if is_analytic_image(file):
                image_data = gd.Open(Path(curr_file).as_posix(), gd.GA_ReadOnly)
                save_image(image_data, Path(root, "plot.png"))
                data = get_band_image(image_data, [2, 4, 6, 7, 8])
                np.save(Path(root, "data.npy"), data)      

def remove_unnecessary_files(month: str, year: str):
    source_dir = Path(f'images/planet_scope/{year}_{month}')
    for root, _, files in os.walk(source_dir):
        for file in files:
            curr_file = Path(root, file)
            if not (file == "plot.png" or file == "data.npy"):
                os.remove(curr_file)

def crop_data(month:str, year:str):
    source_dir = Path(f'images/planet_scope/{year}_{month}')
    for root, _, files in os.walk(source_dir):
        for file in files:
            curr_file = Path(root, file)
            if file == "data.npy":
                file = np.load(curr_file)
                cropped_arr = file[4:671, 4:671, :]
                # np.save(curr_file)

def create_stats(month:str, year: str):
    filename = Path(f'images/planet_scope/stats/{year}_{month}.csv')
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'x') as stats:
        stats.write("name,B2,B4,B6,B7,B8,coordinates")

def write_stats(month:str, year:str):
    points = Path(f'images/coordinates/points.geojson')
    with open(points, 'r') as file:
        print(file)
    # print(points)
    
    # source_dir = Path(f'images/planet_scope/{year}_{month}')
    # for root, _, files in os.walk(source_dir):
    #     for file in files:
    #         curr_file = Path(root, file)
    #         if file == "data.npy":


                



def preprocess_ps_data(month: str, year: str):
    # get_data_and_plot(month, year)
    # remove_unnecessary_files(month, year)
    # crop_data(month, year)
    # create_stats(month, year)
    write_stats(month, year)