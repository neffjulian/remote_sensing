from pathlib import Path

import wget
import py7zr

FILTERED = Path(__file__).parent.parent.joinpath("data", "filtered")
FILTERED_IN_SITU = FILTERED.joinpath("in_situ")

FILTERED.mkdir(exist_ok=True, parents=True)
FILTERED_IN_SITU.mkdir(exist_ok=True, parents=True)

def download_zip_files():
    print("Start downloading Sentinel-2 data...")
    sentinel_url = "https://polybox.ethz.ch/index.php/s/In45d1I7zT6lLxb/download"
    sentinel_in_situ_url = "https://polybox.ethz.ch/index.php/s/MXe8uWICvYRFG7p/download"
    wget.download(sentinel_url,out=str(FILTERED))
    wget.download(sentinel_in_situ_url,out=str(FILTERED_IN_SITU))
    print("\n...finished!")

    print("Start downloading PlanetScope data...")
    planetscope_url = "https://polybox.ethz.ch/index.php/s/zQYzJY4kLNvaxGA/download"
    planetscope_in_situ_url = "https://polybox.ethz.ch/index.php/s/FtplTrxAFY3eOGA/download"

    wget.download(planetscope_url,out=str(FILTERED))
    wget.download(planetscope_in_situ_url,out=str(FILTERED_IN_SITU))

    print("\n...finished!")

def unzip_and_delete(satellite: str):
    print(f"Unzipping {satellite} files...")
    data_zip = FILTERED.joinpath(f"{satellite}.7z")
    in_situ_zip = FILTERED_IN_SITU.joinpath(f"{satellite}_in_situ.7z")

    with py7zr.SevenZipFile(data_zip) as z:
        z.extractall(FILTERED)

    with py7zr.SevenZipFile(in_situ_zip) as z:
        z.extractall(FILTERED_IN_SITU)

    data_zip.unlink()
    in_situ_zip.unlink()
    print("\n...finished!")

if __name__ == "__main__":
    download_zip_files()
    unzip_and_delete("sentinel")
    unzip_and_delete("planetscope")