"""
All code related to downloading the data and model weights.

@date: 2023-08-30
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
from pathlib import Path

import wget
import py7zr

DATA_DIR = Path(__file__).parent.parent.joinpath("data")
WEIGHT_DIR = Path(__file__).parent.parent.joinpath("weights")

def download_data() -> None:
    """
    Downloads the image data from Polybox and extracts it.
    """

    print("Warning: The data necessary for this project is 35 GB in size. Make sure you have enough space on your hard drive and \
          make sure to remove the data in the 'filtered' folder after runnning '/preprocess_data.py'.")
    
    # Download images with wget
    url = "https://polybox.ethz.ch/index.php/s/7xiwwzdKwvy1Ks0/download"
    wget.download(url,out=str(DATA_DIR))

    # Extract the images
    data_zip = DATA_DIR.joinpath("filtered.7z")
    with py7zr.SevenZipFile(data_zip) as z:
        z.extractall(DATA_DIR)
    data_zip.unlink()

def download_model_weights() -> None:
    """
    Downloads model weights for SRCNN, EDSR, and RRDB.
    This allows the user to run the model without having to train it first.
    """

    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download model weights with wget
    print("\n \n Downloading model weights for SRCNN")
    url = "https://polybox.ethz.ch/index.php/s/0KCAl9fFuvc3iz7/download"
    wget.download(url,out=str(WEIGHT_DIR))

    print("\n Downloading model weights for EDSR")
    url = "https://polybox.ethz.ch/index.php/s/AUuIBnTZtWi28Jx/download"
    wget.download(url,out=str(WEIGHT_DIR))

    print("\n Downloading model weights for RRDB")
    url = "https://polybox.ethz.ch/index.php/s/oRn8KSmjPHYP6AJ/download"
    wget.download(url,out=str(WEIGHT_DIR))

    print("\n Downloading model weights for ESRGAN")
    url = "https://polybox.ethz.ch/index.php/s/yTrbpyRglRdcx26/download"
    wget.download(url,out=str(WEIGHT_DIR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=bool)
    parser.add_argument("-w", "--weights", type=bool)

    args = parser.parse_args()

    if args.data is not None:
        parser.data = download_data()

    if args.weights is not None:
        parser.weights = download_model_weights()