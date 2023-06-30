from pathlib import Path
import json

import pyproj
import pandas as pd

from utils import point_to_square

FIELD_DATA = Path(__file__).parent.parent.parent.joinpath("data", "coordinates", "field_data.csv")


def main():
    df = pd.read_csv(FIELD_DATA, usecols=['wkt_geom', 'date', 'lai'])
    data = df.to_numpy()

    source_proj = pyproj.Proj(init='epsg:3857')
    target_proj = pyproj.Proj(init='epsg:4326 ')

    lai_points = {
        "type": "FeatureCollection",
        "features": []
    }

    field_parcels = {
        "type": "FeatureCollection",
        "features": []
    }

    for entry in data:
        point_coords, date, lai = entry
        cleaned_string = point_coords.replace('Point (', '').replace(')', '')
        x, y = cleaned_string.split()
        lon, lat = pyproj.transform(source_proj, target_proj, float(x), float(y))

        feature = {
            "type": "Feature",
            "properties": {
                "date": date,
                "lai": lai
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        }

        field_parcels["features"].append(point_to_square(feature, .5))
        lai_points["features"].append(feature)

    json.dump(lai_points, open(FIELD_DATA.parent.joinpath("field_data.geojson"), "w"))
    json.dump(field_parcels, open(FIELD_DATA.parent.joinpath("field_parcels.geojson"), "w"))

if __name__ == "__main__":
    main()