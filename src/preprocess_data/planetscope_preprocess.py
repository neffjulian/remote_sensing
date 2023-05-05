import os
import argparse
from pathlib import Path

from utils import (
    MONTHS,
    DATA_DIR
)

def preprocess_planetscope_data(year: str, month: str) -> None:
    
    source_dir = DATA_DIR.joinpath("raw", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("preprocessed", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            print(root, filename)

def main(year: str, month: str) -> None:
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    
    preprocess_planetscope_data(year, month)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    main(**vars(args))