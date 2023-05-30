from pathlib import Path

import wget
import py7zr
import zipfile

FILTERED = Path().absolute().parent.parent.joinpath("data", "filtered")

def download_zip_files():
    sentinel_url = "https://polybox.ethz.ch/index.php/s/AdiQZeI9YGmp9Ac/download?path=%2F&files=sentinel.zip"
    planetscope_url = "https://polybox.ethz.ch/index.php/s/AdiQZeI9YGmp9Ac/download?path=%2F&files=planetscope.zip"

    wget.download(sentinel_url,out=str(FILTERED))
    wget.download(planetscope_url,out=str(FILTERED))

def unzip_and_delete():
    sentinel_zip = FILTERED.joinpath("sentinel.zip")
    planetscope_zip = FILTERED.joinpath("planetscope.zip")

    if not sentinel_zip.exists() or not planetscope_zip.exists():
        raise Exception("Copy the files 'planetscope' and 'sentinel' into the 'data/filtered' folder.")
    
    if str(sentinel_zip).endswith(".zip"):
        with zipfile.ZipFile(sentinel_zip, 'r') as zip_ref:
            zip_ref.extractall(FILTERED)
    elif str(sentinel_zip).endswith(".7z"):
        with py7zr.SevenZipFile(sentinel_zip) as z:
            z.extractall(FILTERED)
    else:
        raise Exception("Invalid Format of files.")
    sentinel_zip.unlink()
    
    if str(planetscope_zip).endswith(".zip"):
        with zipfile.ZipFile(planetscope_zip, 'r') as zip_ref:
            zip_ref.extractall(FILTERED)
    elif str(planetscope_zip).endswith(".7z"):
        with py7zr.SevenZipFile(planetscope_zip) as z:
            z.extractall(FILTERED)
    else:
        raise Exception("Invalid Format of files.")
    planetscope_zip.unlink()

if __name__ == "__main__":
    download_zip_files()
    unzip_and_delete()