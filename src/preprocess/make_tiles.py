import os
import argparse
from pathlib import Path

import numpy as np

from utils import (
    DATA_DIR
)

# Iterate through every matching image pair (NDVI).
# Make it to 16 tiles
# Enumerate it {S2/PS}_{year}_{month}_{index}_{tile}
# Save it to tiles
# Make torch Dataset

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
    
def iterate_through_pairs():
    sentinel_dir = DATA_DIR.joinpath("processed", "sentinel")
    planetscope_dir = DATA_DIR.joinpath("processed", "planetscope")

    tiles_sentinel_dir = DATA_DIR.joinpath("tiles", "sentinel")
    tiles_planetscope_dir = DATA_DIR.joinpath("tiles", "planetscope")

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
                    

def main() -> None:
   iterate_through_pairs()

if __name__ == "__main__":
    main()