import argparse

from images.sentinel_2.s2_download import download_s2_data
from images.sentinel_2.s2_preprocess import preprocess_s2_data

from images.planet_scope.ps_download import download_ps_data

# Define a dictionary mapping month names to date ranges
MONTHS = {
    'january': ("01-01", "01-31"),
    'february': ("02-01", "02-28"),
    'march': ("03-01", "03-31"),
    'april': ("04-01", "04-30"),
    'may': ("05-01", "05-31"),
    'june': ("06-01", "06-30"),
    'july': ("07-01", "07-31"),
    'august': ("08-01", "08-31"),
    'september': ("09-01", "09-30"),
    'october': ("10-01", "10-31"),
    'november': ("11-01", "11-30"),
    'december': ("12-01", "12-31"),
}

def main(satellite: str, year: int, month: str) -> None:
    year = year % 100
    if year > 22:
        raise ValueError("Year invalid", year)
    if month.lower() not in MONTHS:
        raise ValueError("Month invalid")
    
    if satellite.startswith("s"):
        download_s2_data(month, str(year))
        preprocess_s2_data(month, str(year))   
    elif satellite.startswith("p"):
        download_ps_data(month, year)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite", required=True, type=str)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    main(**vars(args))