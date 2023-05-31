from pathlib import Path

import numpy as np

from sentinel_preprocess import preprocess_sentinel_data
from planetscope_preprocess import preprocess_planetscope_data, remove_non_image_pairs

from utils import DATA_DIR

def cut_np_array(path: Path, target_dir: Path, year: str, month: str) -> list:
    tiles = 2
    name = f"{year.name[2:]}_{month.name[:2]}_{path.name[:4]}_"

    data = np.load(path)
  
    if data.shape != (192, 192) and data.shape != (648, 648) and data.shape != (96, 96):
        raise Exception(f"Invalid Shape: {data.shape}")

    x, y = data.shape
    new_x, new_y = int(x/tiles), int(y/tiles)
    sub_data = []

    for i in range(tiles):
        for j in range(tiles):
            sub_data.append(data[(i * new_x):((i+1) * new_x), (j * new_y):((j+1) * new_y)])

    for index, tile in enumerate(sub_data):
        filename = name + f"{index:01d}"
        np.save(target_dir.joinpath(filename), tile)
    
def create_tiles():
    print("Started creating tiles!")
    sentinel_dir = DATA_DIR.joinpath("processed", "sentinel")

    tiles_sentinel_10m_dir = DATA_DIR.joinpath("tiles", "sentinel_10m")
    tiles_sentinel_20m_dir = DATA_DIR.joinpath("tiles", "sentinel_20m")
    tiles_planetscope_4b_dir = DATA_DIR.joinpath("tiles", "planetscope_4b")
    tiles_planetscope_8b_dir = DATA_DIR.joinpath("tiles", "planetscope_8b")

    tiles_sentinel_10m_dir.mkdir(parents=True, exist_ok=True)
    tiles_sentinel_20m_dir.mkdir(parents=True, exist_ok=True)
    tiles_planetscope_4b_dir.mkdir(parents=True, exist_ok=True)
    tiles_planetscope_8b_dir.mkdir(parents=True, exist_ok=True)

    for year in sentinel_dir.iterdir():
        for month in year.iterdir():
            lai_folder = month.joinpath("lai")
            for sentinel_file in lai_folder.iterdir():
                if sentinel_file.name.endswith("10m.npy"):
                    sentinel_10m = sentinel_file
                    sentinel_20m = Path(str(sentinel_file).replace("10m", "20m"))
                    planetscope = Path(str(sentinel_file).replace("sentinel", "planetscope"))
                    planetscope_4b = Path(str(planetscope).replace("10m", "4bands"))
                    planetscope_8b = Path(str(planetscope).replace("10m", "8bands"))

                    cut_np_array(sentinel_10m, tiles_sentinel_10m_dir, year, month)
                    cut_np_array(sentinel_20m, tiles_sentinel_20m_dir, year, month)
                    cut_np_array(planetscope_4b, tiles_planetscope_4b_dir, year, month)
                    cut_np_array(planetscope_8b, tiles_planetscope_8b_dir, year, month)
    
    print("Finished creating tiles!")
                    
def preprocess_data(year: str, month: str):
    print(f"Preprocess data for {year} {month}: Started")
    preprocess_sentinel_data(year, month)
    preprocess_planetscope_data(year, month)
    remove_non_image_pairs(year, month)
    print(f"Preprocess data for {year} {month}: Finished")

def main():
    preprocess_data("2022", "apr")
    preprocess_data("2022", "may")
    preprocess_data("2022", "jun")
    preprocess_data("2022", "sep")
    create_tiles()

if __name__ == "__main__":
    main()