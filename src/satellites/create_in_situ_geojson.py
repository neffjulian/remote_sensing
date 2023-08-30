"""
Read field data from a CSV file, transform coordinates, and save as GeoJSON.

@date: 2023-08-30
@author: Julian Neff, ETH Zurich

Copyright (C) 2023 Julian Neff

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from pathlib import Path
import json

import pyproj
import pandas as pd

from utils import point_to_square

FIELD_DATA = Path(__file__).parent.parent.parent.joinpath("data", "coordinates", "field_data.csv")

def main():
    """
    This function reads field data from a CSV file, transforms the coordinates,
    and generates two GeoJSON files: one for LAI points and another for field parcels.
    """
    
    # Read CSV and select specific columns
    df = pd.read_csv(FIELD_DATA, usecols=['wkt_geom', 'date', 'lai'])
    data = df.to_numpy()

    # Define coordinate projection systems
    source_proj = pyproj.Proj(init='epsg:3857')
    target_proj = pyproj.Proj(init='epsg:4326')

    # Initialize dictionaries for LAI points and field parcels
    lai_points = {
        "type": "FeatureCollection",
        "features": []
    }

    field_parcels = {
        "type": "FeatureCollection",
        "features": []
    }

    # Iterate through each entry in the data
    for entry in data:
        point_coords, date, lai = entry
        cleaned_string = point_coords.replace('Point (', '').replace(')', '')
        x, y = cleaned_string.split()
        lon, lat = pyproj.transform(source_proj, target_proj, float(x), float(y))

        # Create a GeoJSON feature for LAI points
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

        # Convert the point feature to a square polygon and add to field parcels
        field_parcels["features"].append(point_to_square(feature, 0.5))
        lai_points["features"].append(feature)

    # Write the GeoJSON data to files
    with open(FIELD_DATA.parent.joinpath("field_data.geojson"), "w") as lai_file:
        json.dump(lai_points, lai_file)

    with open(FIELD_DATA.parent.joinpath("field_parcels.geojson"), "w") as parcels_file:
        json.dump(field_parcels, parcels_file)

if __name__ == "__main__":
    main()