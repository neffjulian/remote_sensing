import argparse

from images.sentinel_2.s2_download import download_s2_data
from images.sentinel_2.s2_preprocess import preprocess_s2_data

from images.planet_scope.ps_download import download_ps_data
from images.planet_scope.ps_preprocess import preprocess_ps_data

from images.swissimage.si_download import download_si_data

# Define a dictionary mapping month names to date ranges
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", \
          "jul", "aug", "sep", "oct", "nov", "dec"]

def main(satellite: str, year: int, month: str) -> None:
    if not(0 < year or year < 22):
        raise ValueError(f"Year invalid ({year})")
    if month not in MONTHS:
        raise ValueError(f"Month invalid ({month})")
    
    # TODO: change to more cleaner method
    if satellite.startswith("se"):
        download_s2_data(month, year)
        preprocess_s2_data(month, year)   

    elif satellite.startswith("p"):
        download_ps_data(month, year)
        preprocess_ps_data(month, year)
        
    elif satellite.startswith("sw"):
        download_si_data(month, year)
    else:
        raise ValueError(f"Satellite invalid ({satellite})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite", required=True, type=str)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    # main(**vars(args))

    for month in MONTHS:
        main("sentinel", 22, month)