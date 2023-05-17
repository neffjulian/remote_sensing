from pathlib import Path

import wget

FILTERED = Path().absolute().parent.parent.joinpath("data", "filtered")

def main():
    sentinel_url = "https://polybox.ethz.ch/index.php/s/f3A3sP40G3MKvBJ/download"
    planetscope_url = "https://polybox.ethz.ch/index.php/s/1n5zC3CZGd4ECBQ/download"
    wget.download(sentinel_url,out=str(FILTERED))
    wget.download(planetscope_url,out=str(FILTERED))

if __name__ == "__main__":
    main()