import os
from pathlib import Path


import numpy as np
import osgeo.gdal as gd
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_si_data(month: str, year: str):
    data_dir = Path(f'images/swissimage/{year}_{month}')

    if not data_dir.exists():
        raise Exception(f"No Data available for {year} {month}. Check if the folder {data_dir}.")
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.tif'):
                curr_file = Path(root, file)
                image = Image.open(curr_file)
                data = np.array(image)
                np.save(Path(root, 'array.npy'), data)
                plt.imshow(data)
                plt.axis('off')
                plt.savefig(Path(root, 'plot.png'))