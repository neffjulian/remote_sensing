import argparse

from images.sentinel_2.s2_download import download_s2_data
# from images.sentinel_2.s2_preprocess import preprocess_s2_data

# from images.planet_scope.ps_download import place_ps_order, download_ps_data
# from images.planet_scope.ps_preprocess import preprocess_ps_data
# from images.planet_scope.ps_sample_download import test

# from images.swissimage.si_download import download_si_data
# from images.swissimage.si_preprocess import preprocess_si_data

# Define a dictionary mapping month names to date ranges
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", \
          "jul", "aug", "sep", "oct", "nov", "dec"]

def main(satellite: str, year: str, month: str) -> None:
    if not(2016 < year and year < 2023):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022' as the data is only available in this range.")
    if month not in MONTHS:
        raise ValueError(f"Month invalid ({month}). Use one out of {MONTHS}.")
    
    if satellite == "sentinel":
        download_s2_data(year, month)
        # preprocess_s2_data(month, year)   

    elif satellite == "planetscope":
        pass
        # place_ps_order(month, year)
        # download_ps_data(month, year)
        # preprocess_ps_data(month, year)
    elif satellite == "swissimage":
        pass
        # download_si_data(month, year)
        # preprocess_si_data(month, year)
    else:
        raise ValueError(f"Satellite invalid ({satellite}). Either use 'sentinel', 'planetscope' or 'swissimage'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite", required=True, type=str)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    main(**vars(args))

    # for month in MONTHS:
    #     main("sentinel", 22, month)