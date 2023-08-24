from pathlib import Path
import json
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent.joinpath("data")
MONTH_MAPS = {"March": "03_mar", "April": "04_apr", "May": "05_may", "June": "06_jun", "July": "07_jul", "August": "08_aug", "September": "09_sep"}

def get_planetscope_date(index: int, month: str):
    data = DATA_DIR.joinpath("filtered", "planetscope", "2022", MONTH_MAPS[month], "lai", f"{index:04d}_lai_4bands.tif")
    file = DATA_DIR.joinpath("filtered", "planetscope", "2022", MONTH_MAPS[month], "metadata", f"{index:04d}.xml")
    if file.exists() and data.exists():
        tree = ET.parse(file)
        root = tree.getroot()
        return root[0][0][4][0][1].text
    else:
        return "-"

def get_sentinel_date(index: int, month: str):
    data = DATA_DIR.joinpath("filtered", "sentinel", "2022", MONTH_MAPS[month], "lai", f"{index:04d}_scene_10m_lai.tif")
    file = DATA_DIR.joinpath("filtered", "sentinel", "2022", MONTH_MAPS[month], "metadata", f"{index:04d}_MTD_TL.xml")
    if file.exists() and data.exists():
        tree = ET.parse(file)
        root = tree.getroot()
        return root[0][4].text
    else:
        # print(index, month, "sentinel")
        return "-"
    
def parse_timestamp(timestamp_str):
    try:
        # Attempt to parse the timestamp
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        return datetime.fromisoformat(timestamp_str)
    except ValueError:
        # Return None if parsing fails
        return None
    
def get_time_diff(timestamp1: str, timestamp2: str):
    dt1 = parse_timestamp(timestamp1)
    dt2 = parse_timestamp(timestamp2)

    if dt1 is None or dt2 is None:
        return "-"
    
    difference = dt2 - dt1
    days = difference.days
    hours, remainder = divmod(difference.seconds, 3600)
    return hours
    
def verify_files():
    points_ch = DATA_DIR.joinpath("coordinates", "points_ch.geojson")
    with open(points_ch) as f:
        points = json.load(f)
    
    # create dataframe with header ("month", "index", "latitude", "longitude", "planetscope_capture_date", "sentinel_capture_date", "area")
    df = pd.DataFrame(index = range(2072), columns=["month", "index", "latitude", "longitude", "planetscope_capture_date", "sentinel_capture_date", "difference_hours", "area_sq_km"])
    for i, month in enumerate(list(MONTH_MAPS.keys())):
        for index in range(296):
            df.iloc[i*296 + index]["month"] = month
            df.iloc[i*296 + index]["index"] = index
            df.iloc[i*296 + index]["latitude"] = points["features"][index]["geometry"]["coordinates"][1]
            df.iloc[i*296 + index]["longitude"] = points["features"][index]["geometry"]["coordinates"][0]
            s2_time = get_sentinel_date(index, month)
            ps_time = get_planetscope_date(index, month)
            df.iloc[i*296 + index]["sentinel_capture_date"] = s2_time
            df.iloc[i*296 + index]["planetscope_capture_date"] = ps_time
            df.iloc[i*296 + index]["difference_hours"] = get_time_diff(s2_time, ps_time)

            if s2_time != "-" and ps_time != "-":
                df.iloc[i*296 + index]["area_sq_km"] = 4

    df.to_csv(DATA_DIR.joinpath("filtered", "verify.csv"), index=False)

def main():
    verify_files()



if __name__ == "__main__":
    main()