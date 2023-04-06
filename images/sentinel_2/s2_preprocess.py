import os
import json
import csv
import numpy as np

# Define the bands to retrieve
bands = ["B02", "B03", "B04", "B05", "B8A"]

def get_coordinates_from_points(filepath: str) -> list:
    """
    Retrieve a list of coordinates from a GeoJSON file and return them.

    Parameters:
    filepath (str): The filepath of the GeoJSON file.

    Returns:
    list: A list of coordinates.
    """
    with open(filepath, 'r') as file:
        points = json.load(file)

    coordinates = []
    for point in points['features']:
        point_coords = point['geometry']['coordinates']
        coordinates.append(point_coords)

    return coordinates

def update_bounds(arr: np.ndarray, low: int, high: int) -> tuple:
    """
    Update the bounds of an array. Used to check that image sizes are consistent and remove outliers from the downloaded data.

    Parameters:
    arr (numpy.ndarray): The input array.
    low (int): The current lower bound.
    high (int): The current upper bound.

    Returns:
    tuple: A tuple containing the updated lower and upper bounds.
    """

    shape = np.shape(arr)
    low = min(low, min(shape[1], shape[2]))
    high = max(high, max(shape[1], shape[2]))
    return low, high

def remove_files(name: str, filename: str) -> None:
    """
    Remove a .npz file and its corresponding .png file. This function is called, when the downloaded file is really small or has an unreadable part.

    Parameters:
    name (str): The name of the directory where the files are located.
    filename (str): The name of the .npz file to be removed.
    """

    os.remove(f"images/sentinel_2/{name}/{filename}")
    os.remove(f"images/sentinel_2/{name}/{filename[:-4]}.png")

def crop(arr: np.ndarray, z: int) -> np.ndarray:
    """
    Crop an array to a specified size. Keeps the center of the array.

    Parameters:
    arr (numpy.ndarray): The input array to be cropped.
    z (int): The size to which the array will be cropped.

    Returns:
    numpy.ndarray: The cropped array.
    """

    _, x, y = arr.shape
    start_x = (x - z) // 2
    start_y = (y - z) // 2
    
    end_x = start_x + z
    end_y = start_y + z
    
    new_arr = np.zeros((1, z, z))
    new_arr[0,:,:] = arr[0,start_x:end_x,start_y:end_y]
    return new_arr

def verify_dimensions(name) -> None:
    """
    Verify that the dimensions of the Sentinel-2 data are consistent and remove outliers.

    Parameters:
    name (str): The name of the directory where the .npz files are located.
    """

    for filename in os.listdir(f'images/sentinel_2/{name}'):
        if filename.endswith('.npz'):
            file = dict(np.load(f'images/sentinel_2/{name}/{filename}'))

            for key in file.keys():
                shape = np.shape(file[key])
                len = min(shape[1], shape[2])
                if key in ['B02', 'B03', 'B04']:
                    if(len < 200):
                        remove_files(name, filename)
                        break
                elif key in ['B05', 'B8A']:
                    if(len < 100):
                        remove_files(name, filename)
                        break
                else:
                    raise Exception("Invalid key")

def crop_data(name: str, min_10: int, min_20: int) -> None:
    """
    Crop the arrays in .npz files in the given directory to the specified dimensions based on their bands.

    Parameters:
    name (str): Name of the folder containing the data
    min_10 (int): The minimum size of bands with 10m spatial resolution
    min_20 (int): The minimum size of bands with 20m spatial resolution
    """

    for filename in os.listdir(f'images/sentinel_2/{name}'):
        if filename.endswith('.npz'):
            band_data = []
            file = dict(np.load(f'images/sentinel_2/{name}/' + filename))
            for key in file.keys():
                if key in ['B02', 'B03', 'B04']:
                    band_data.append(crop(file[key], min_10))
                elif key in ['B05', 'B8A']:
                    band_data.append(crop(file[key], min_20))
                else:
                    raise Exception("Invalid key")
                
            np.savez(
            f"images/sentinel_2/{name}/{filename}",
            **{band: data for band, data in zip(bands, band_data)}
            )

def create_stats(foldername: str) -> None:
    """
    Create a new statistics CSV file for the given folder name or remove and replace the existing file.

    Parameters:
    foldername (str): Name of the folder containing the data
    """
    if os.path.exists(f'images/sentinel_2/stats/{foldername}_stats.csv'):
        os.remove(f'images/sentinel_2/stats/{foldername}_stats.csv')
    with open(f'images/sentinel_2/stats/{foldername}_stats.csv', 'x') as file:
        file.write("name,B02,B03,B04,B05,B8A,coordinates \n")

def write_stats_to_csv(name: str, band_data: np.ndarray, point: tuple, foldername: str) -> None:
    """
    Write the statistics for a given file to the CSV file for the given folder name.

    Parameters:
    name (str): Name of the array containing the data
    band_data (np.ndarray): Numpy array containing the relevant data
    point (tuple): Center coordinates of the corresponding area which band_data covers,
    foldername (str): Name of the folder containing the data
    """
    with open(f'images/sentinel_2/stats/{foldername}_stats.csv', mode="a", newline="") as stats:
        writer = csv.writer(stats)
        row = [name] + [np.shape(band_data[band]) for band in bands] + [point]
        writer.writerow(row)

def verify_data(name: str) -> None:
    """
    Verify the dimensions of the arrays in .npz files in the given directory and write statistics to a CSV file.

    Parameters:
    name (str): Name of the folder containing the data
    """
    create_stats(name)
    point_coordinates = get_coordinates_from_points('images/coordinates/points.geojson')

    for filename in os.listdir(f'images/sentinel_2/{name}'):
        if filename.endswith('.npz'):
            number = int(filename[:-4])
            file = np.load(f'images/sentinel_2/{name}/' + filename)
            write_stats_to_csv(number, file, point_coordinates[number], name)

def preprocess_s2_data(month: str, year: str) -> None:
    """
    Preprocess the data in the given directory by verifying dimensions, cropping the arrays, and writing statistics.

    Args:
        month (str): Name of the month the data should be collected from
        year (str): The year during which the data should be collected from

    """
    name = f"{year[-2:]}_{month}"
    
    verify_dimensions(name)
    crop_data(name, 200, 100)
    verify_data(name)