import argparse
from pathlib import Path

import wget
import py7zr

DATA_DIR = Path(__file__).parent.parent.joinpath("data")
WEIGHT_DIR = Path(__file__).parent.parent.joinpath("weights")

def download_data() -> None:
    print("Warning: The data necessary for this project is 35 GB in size. Make sure you have enough space on your hard drive and \
          make sure to remove the data in the 'filtered' folder after runnning '/preprocess_data.py'.")
    
    url = "https://polybox.ethz.ch/index.php/s/7xiwwzdKwvy1Ks0/download"
    wget.download(url,out=str(DATA_DIR))

    data_zip = DATA_DIR.joinpath("filtered.7z")
    with py7zr.SevenZipFile(data_zip) as z:
        z.extractall(DATA_DIR)
    data_zip.unlink()

def download_model_weights() -> None:
    print("\n \n Downloading model weights for SRCNN")
    url = "https://polybox.ethz.ch/index.php/s/kNEZ2m5ZOjOBdDq/download"
    wget.download(url,out=str(WEIGHT_DIR))

    print("\n Downloading model weights for EDSR")
    url = "https://polybox.ethz.ch/index.php/s/AwqJ7rCallI147h/download"
    wget.download(url,out=str(WEIGHT_DIR))

    print("\n Downloading model weights for RRDB")
    url = "https://polybox.ethz.ch/index.php/s/h4lbEodffke5Eru/download"
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