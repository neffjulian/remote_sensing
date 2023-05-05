import os
import argparse
from pathlib import Path

from distutils.dir_util import copy_tree

from utils import (
    MONTHS,
    DATA_DIR
)

def preprocess_sentinel_data(year: str, month: str) -> None:
    source_dir = DATA_DIR.joinpath("raw", "sentinel", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("preprocessed", "sentinel", year, f"{MONTHS[month]}_{month}", "data")
    # target_dir.mkdir(parents=True, exist_ok=True)
    copy_tree(source_dir.as_posix(), target_dir.as_posix())


def main(year: str, month: str) -> None:
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    
    preprocess_sentinel_data(year, month)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    main(**vars(args))