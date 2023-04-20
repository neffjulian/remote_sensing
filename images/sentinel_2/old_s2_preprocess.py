import os
import json
import csv
import numpy as np

# Define the bands to retrieve
bands = ["B02", "B03", "B04", "B05", "B8A"]

def get_coordinates_from_points(filepath: str) -> list:
    with open(filepath, 'r') as file:
        points = json.load(file)

    coordinates = []
    for point in points['features']:
        point_coords = point['geometry']['coordinates']
        coordinates.append(point_coords)

    return coordinates

def update_bounds(arr: np.ndarray, low: int, high: int) -> tuple:
    shape = np.shape(arr)
    low = min(low, min(shape[1], shape[2]))
    high = max(high, max(shape[1], shape[2]))
    return low, high

def remove_files(name: str, filename: str) -> None:
    os.remove(f"images/sentinel_2/{name}/{filename}")
    os.remove(f"images/sentinel_2/{name}/{filename[:-4]}.png")

def crop(arr: np.ndarray, z: int) -> np.ndarray:
    _, x, y = arr.shape
    start_x = (x - z) // 2
    start_y = (y - z) // 2
    
    end_x = start_x + z
    end_y = start_y + z
    
    new_arr = np.zeros((1, z, z))
    new_arr[0,:,:] = arr[0,start_x:end_x,start_y:end_y]
    return new_arr

def verify_dimensions(name) -> None:
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
    if os.path.exists(f'images/sentinel_2/stats/{foldername}_stats.csv'):
        os.remove(f'images/sentinel_2/stats/{foldername}_stats.csv')
    with open(f'images/sentinel_2/stats/{foldername}_stats.csv', 'x') as file:
        file.write("name,B02,B03,B04,B05,B8A,coordinates \n")

def write_stats_to_csv(name: str, band_data: np.ndarray, point: tuple, foldername: str) -> None:
    with open(f'images/sentinel_2/stats/{foldername}_stats.csv', mode="a", newline="") as stats:
        writer = csv.writer(stats)
        row = [name] + [np.shape(band_data[band]) for band in bands] + [point]
        writer.writerow(row)

def verify_data(name: str) -> None:
    create_stats(name)
    point_coordinates = get_coordinates_from_points('images/coordinates/points.geojson')

    for filename in os.listdir(f'images/sentinel_2/{name}'):
        if filename.endswith('.npz'):
            number = int(filename[:-4])
            file = np.load(f'images/sentinel_2/{name}/' + filename)
            write_stats_to_csv(number, file, point_coordinates[number], name)

def preprocess_s2_data(month: str, year: str) -> None:
    name = f"{year[-2:]}_{month}"
    
    verify_dimensions(name)
    crop_data(name, 200, 100)
    verify_data(name)