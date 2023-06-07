from pathlib import Path

import wget
import py7zr

FILTERED = Path().absolute().parent.parent.joinpath("data", "filtered")
FILTERED_IN_SITU = FILTERED.joinpath("in_situ")

def download_zip_files():
    sentinel_url = "https://polybox.ethz.ch/index.php/s/In45d1I7zT6lLxb/download"
    sentinel_in_situ_url = "https://polybox.ethz.ch/index.php/s/9CWEUvvkCEPIDsX/download"
    wget.download(sentinel_url,out=str(FILTERED))
    wget.download(sentinel_in_situ_url,out=str(FILTERED_IN_SITU))

    planetscope_url = "https://polybox.ethz.ch/index.php/s/Tvj5Au8bg7cjnZX/download"
    planetscope_in_situ_url = "https://polybox.ethz.ch/index.php/s/jFtT0TarzqcNHr8/download"

    wget.download(planetscope_url,out=str(FILTERED))
    wget.download(planetscope_in_situ_url,out=str(FILTERED_IN_SITU))

def unzip_and_delete(satellite: str):
    data_zip = FILTERED.joinpath(f"{satellite}.7z")
    in_situ_zip = FILTERED_IN_SITU.joinpath(f"{satellite}_in_situ.7z")

    with py7zr.SevenZipFile(data_zip) as z:
        z.extractall(FILTERED)

    with py7zr.SevenZipFile(in_situ_zip) as z:
        z.extractall(FILTERED_IN_SITU)

    data_zip.unling()
    in_situ_zip.unlink()

if __name__ == "__main__":
    download_zip_files()
    unzip_and_delete("sentinel")
    unzip_and_delete("planetscope")