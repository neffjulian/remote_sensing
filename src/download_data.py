from pathlib import Path

import wget
import py7zr

DATA_DIR = Path(__file__).parent.parent.joinpath("data")

def download_zip_files():
    url = "https://polybox.ethz.ch/index.php/s/e5yrmHcejuCo0WO/download"
    wget.download(url,out=str(DATA_DIR))

def unzip_and_delete():
    data_zip = DATA_DIR.joinpath("filtered.7z")

    with py7zr.SevenZipFile(data_zip) as z:
        z.extractall(DATA_DIR)

    data_zip.unlink()

if __name__ == "__main__":
    download_zip_files()
    unzip_and_delete()