from pathlib import Path

import wget
import py7zr

DATA_DIR = Path(__file__).parent.parent.joinpath("data")
SENTINEL_IN_SITU = DATA_DIR.joinpath("filtered", "in_situ")

def download_zip_files():
    url = "https://polybox.ethz.ch/index.php/s/e5yrmHcejuCo0WO/download"
    # sentinel_in_situ_url = "https://polybox.ethz.ch/index.php/s/Wq2qZ4ZOw9K9Qyy/download"
    planetscope_in_situ_url = "https://polybox.ethz.ch/index.php/s/d2wdH4nh37fP2Pj/download"
    # wget.download(url,out=str(DATA_DIR))
    # wget.download(sentinel_in_situ_url,out=str(SENTINEL_IN_SITU))
    wget.download(planetscope_in_situ_url,out=str(SENTINEL_IN_SITU))

def unzip_and_delete():
    data_zip = DATA_DIR.joinpath("filtered.7z")
    # sentinel_in_situ_zip = SENTINEL_IN_SITU.joinpath("sentinel_in_situ.7z")
    planetscope_in_situ_zip = SENTINEL_IN_SITU.joinpath("planetscope_in_situ.7z")
    # with py7zr.SevenZipFile(data_zip) as z:
    #     z.extractall(DATA_DIR)

    # with py7zr.SevenZipFile(sentinel_in_situ_zip) as z:
    #     z.extractall(SENTINEL_IN_SITU)

    with py7zr.SevenZipFile(planetscope_in_situ_zip) as z:
        z.extractall(SENTINEL_IN_SITU)

    sentinel_in_situ_zip.unlink()
    planetscope_in_situ_zip.unlink()
    # data_zip.unlink()

if __name__ == "__main__":
    download_zip_files()
    unzip_and_delete()