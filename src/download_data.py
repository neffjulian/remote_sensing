from pathlib import Path

import wget
import py7zr

DATA_DIR = Path(__file__).parent.parent.joinpath("data")

if __name__ == "__main__":
    url = "https://polybox.ethz.ch/index.php/s/7xiwwzdKwvy1Ks0/download"
    wget.download(url,out=str(DATA_DIR))

    data_zip = DATA_DIR.joinpath("filtered.7z")
    with py7zr.SevenZipFile(data_zip) as z:
        z.extractall(DATA_DIR)
    data_zip.unlink()