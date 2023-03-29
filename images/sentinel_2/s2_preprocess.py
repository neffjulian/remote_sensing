import os
import json
import csv
import numpy as np

# Define the bands to retrieve
bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12",]

def get_coordinates_from_points():
    with open('../coordinates/points.geojson', 'r') as file:
        points = json.load(file)

    coordinates = []
    for point in points['features']:
        point_coords = point['geometry']['coordinates']
        coordinates.append(point_coords)

    return coordinates

def update_bounds(arr, low, high):
    shape = np.shape(arr)
    low = min(low, min(shape[1], shape[2]))
    high = max(high, max(shape[1], shape[2]))
    return low, high

def remove_files(name, filename):
    os.remove(f"{name}/{filename}")
    os.remove(f"{name}/{filename[:-4]}.png")

def crop(arr, z):
    _, x, y = arr.shape
    start_x = (x - z) // 2
    start_y = (y - z) // 2
    
    end_x = start_x + z
    end_y = start_y + z
    
    new_arr = np.zeros((1, z, z))
    new_arr[0,:,:] = arr[0,start_x:end_x,start_y:end_y]
    return new_arr

def verify_dimensions(name):
    for filename in os.listdir(f'{name}'):
        if filename.endswith('.npz'):
            file = dict(np.load(f'{name}/' + filename))

            for key in file.keys():
                shape = np.shape(file[key])
                len = min(shape[1], shape[2])
                if key in ['B01', 'B09']:
                    if(len < 33):
                        print(filename, ": Should 33 is ", len)
                        remove_files(name, filename)
                        break
                elif key in ['B02', 'B03', 'B04', 'B08']:
                    if(len < 200):
                        print(filename, ": Should 200 is ", len)
                        remove_files(name, filename)
                        break
                elif key in ['B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12']:
                    if(len < 100):
                        print(filename, ": Should 100 is ", len)
                        remove_files(name, filename)
                        break
                else:
                    raise Exception("Invalid key")

def crop_data(name, min_10, min_20, min_60):

    for filename in os.listdir(f'{name}'):
        if filename.endswith('.npz'):
            band_data = []
            file = dict(np.load(f'{name}/' + filename))
            for key in file.keys():
                if key in ['B01', 'B09']:
                    band_data.append(crop(file[key], min_60))
                elif key in ['B02', 'B03', 'B04', 'B08']:
                    band_data.append(crop(file[key], min_10))
                elif key in ['B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12']:
                    band_data.append(crop(file[key], min_20))
                else:
                    raise Exception("Invalid key")
                
            np.savez(
            f"{name}/{filename}",
            **{band: data for band, data in zip(bands, band_data)}
            )

def create_stats(foldername):
    if os.path.exists(f'stats/{foldername}_stats.csv'):
        os.remove(f'stats/{foldername}_stats.csv')
    with open(f'stats/{foldername}_stats.csv', 'x') as file:
        file.write("name,B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12,coordinates \n")

def write_stats_to_csv(name, band_data, point, foldername):
    with open(f'stats/{foldername}_stats.csv', mode="a", newline="") as stats:
        writer = csv.writer(stats)
        row = [name] + [np.shape(band_data[band]) for band in bands] + [point]
        writer.writerow(row)

def verify_data(name):
    create_stats(name)
    point_coordinates = get_coordinates_from_points()

    for filename in os.listdir(f'{name}'):
        if filename.endswith('.npz'):
            number = int(filename[:-4])
            file = np.load(f'{name}/' + filename)
            write_stats_to_csv(number, file, point_coordinates[number], name)

def preprocess_data(name):
    verify_dimensions(name)
    crop_data(name, 200, 100, 33)
    verify_data(name)