import argparse

from download_data.sentinel2_download import download_sentinel_data

from download_data.planetscope_download import place_planetscope_orders, download_planetscope_orders

from download_data.eschikon_download import download_eschikon_data

from download_data.utils import MONTHS

def main(satellite: str, year: str, month: str, test: bool):
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    

    if test is True:
        coordinate_file = 'point_ai.geojson'
    else:
        coordinate_file = 'points_ch.geojson'


    if satellite == "sentinel":
        download_sentinel_data(coordinate_file, year, month)

    elif satellite == "planetscope":
        place_planetscope_orders(coordinate_file, year, month)
        # download_planetscope_orders(year, month)
    elif satellite == "eschikon":
        download_eschikon_data(year, month)

    else:
        raise ValueError(f"Satellite invalid ('{satellite}'). Either use 'sentinel' or 'planetscope'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite", required=True, type=str)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--month", required=True, type=str)
    parser.add_argument("--test", type=bool)

    args = parser.parse_args()
    main(**vars(args))