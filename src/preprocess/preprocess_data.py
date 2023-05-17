import py7zr

from sentinel_preprocess import preprocess_sentinel_data
from planetscope_preprocess import preprocess_planetscope_data, remove_non_image_pairs

from utils import DATA_DIR

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

if __name__ == "__main__":
    main()