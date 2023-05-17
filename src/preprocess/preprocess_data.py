from pathlib import Path

import py7zr
import numpy as np

from sentinel_preprocess import preprocess_sentinel_data
from planetscope_preprocess import preprocess_planetscope_data, remove_non_image_pairs

from utils import DATA_DIR

def cut_np_array(path: Path, target_dir: Path, year: str, month: str) -> list:
    tiles = 4
    name = f"{year.name[2:]}_{month.name[:2]}_{path.name[5:9]}_"

    data = np.load(path)
    x, y = data.shape
    
    if data.shape != (192, 192) and data.shape != (648, 648):
        raise Exception("Invalid Shape")

    new_x, new_y = int(x/tiles), int(y/tiles)
    sub_data = []

    for i in range(tiles):
        for j in range(tiles):
            sub_data.append(data[(i * new_x):((i+1) * new_x), (j * new_y):((j+1) * new_y)])

    for index, tile in enumerate(sub_data):
        filename = name + f"{index:02d}"
        np.save(target_dir.joinpath(filename), tile)
    
def create_tiles():
    sentinel_dir = DATA_DIR.joinpath("processed", "sentinel")
    planetscope_dir = DATA_DIR.joinpath("processed", "planetscope")

    tiles_sentinel_dir = DATA_DIR.joinpath("tiles", "sentinel")
    tiles_planetscope_dir = DATA_DIR.joinpath("tiles", "planetscope")

    tiles_sentinel_dir.mkdir(parents=True)
    tiles_planetscope_dir.mkdir(parents=True)

    for year in sentinel_dir.iterdir():
        for month in year.iterdir():
            for sentinel_file in month.joinpath("data").iterdir():
                if "ndvi" in sentinel_file.name:
                    planetscope_file = Path(str(sentinel_file).replace("sentinel", "planetscope"))

                    if not planetscope_file.exists():
                        raise Exception("No matching Planetscope file", planetscope_file)
                    
                    if not sentinel_file.name[5:9].isdigit():
                        raise Exception("File has wrong format")
                    
                    cut_np_array(sentinel_file, tiles_sentinel_dir, year, month)
                    cut_np_array(planetscope_file, tiles_planetscope_dir, year, month)

def unzip_and_delete():
    filtered = DATA_DIR.joinpath("filtered")
    sentinel_zip = filtered.joinpath("sentinel.7z")
    planetscope_zip = filtered.joinpath("planetscope.7z")

    if not sentinel_zip.exists() or not planetscope_zip.exists():
        raise Exception("Copy the files 'planetscope.7z' and 'sentinel.7z' into the 'data/filtered' folder.")

    with py7zr.SevenZipFile(sentinel_zip) as z:
        z.extractall(filtered)
    sentinel_zip.unlink()

    with py7zr.SevenZipFile(planetscope_zip) as z:
        z.extractall(filtered)
    planetscope_zip.unlink()

def preprocess_data(year: str, month: str):
    preprocess_sentinel_data(year, month)
    preprocess_planetscope_data(year, month)
    remove_non_image_pairs(year, month)

def main():
    unzip_and_delete()
    preprocess_data("2022", "apr")
    preprocess_data("2022", "may")
    preprocess_data("2022", "jun")
    preprocess_data("2022", "sep")
    create_tiles()

if __name__ == "__main__":
    main()