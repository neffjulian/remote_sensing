"""
Generates Areas of Interest (AOI) based on Green Leaf Area Index (GLAI) rasters from the LAI4SR project. 
Intended for validating super-resolved GLAI products like those from Sentinel-2.

Note: Code directly copied and minorly adopted from https://github.com/lukasValentin/field_boundaries/blob/master/src/field_bounaries.py

@date: 2023-08-28
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

import warnings
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from eodal.core.band import Band
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import Polygon

warnings.filterwarnings('ignore')

DIST_BETWEEN_PARCELS = 10  # meters
VALIDATE_DIR = Path(__file__).parent.parent.parent.joinpath("data", "validate")

def buffer_geom(geom: Polygon) -> Polygon:
    """
    Buffer a geometry by DIST_BETWEEN_PARCELS meters.

    :param geom: Geometry to buffer.
    :return: Buffered geometry.
    """
    _geom = deepcopy(geom)
    return _geom.buffer(DIST_BETWEEN_PARCELS)

def identify_boundaries_with_gradient(
    field_bounaries_path: Path
) -> gpd.GeoDataFrame:
    """
    Identify field boundaries where there is a sharp gradient in median GLAI
    values.

    :param field_bounaries_path: Path to the field boundaries.
    :return: GeoDataFrame with field boundaries where there is a sharp gradient
        in median GLAI values.
    """
    # read field boundaries
    field_boundaries = gpd.read_file(field_bounaries_path)
    # drop empty records
    field_boundaries.dropna(inplace=True, subset=['median'])

    # loop through field boundaries. For each field, look at
    # the neigbors and store the largest gradient in field median GLAI
    # values
    field_boundaries['max_gradient'] = None
    field_boundaries['neigbor_geometry'] = None

    for idx, row in field_boundaries.iterrows():
        # construct a buffer of DIST_BETWEEN_PARCELS meters around the field
        # boundary to define the neighborhood
        buffered_reference_geom = buffer_geom(row.geometry)

        # get neighbors
        neighbors = field_boundaries[
            field_boundaries.geometry.overlaps(buffered_reference_geom)].copy()
        # if there are no neighbors, continue
        if neighbors.empty:
            continue
        # compute gradient in absolute values
        gradient = abs(neighbors['median'].max() - row['median'])
        # store gradient
        field_boundaries.loc[idx, 'max_gradient'] = gradient
        # store the geometry of the neighbor with the largest gradient
        field_boundaries.loc[idx, 'neigbor_geometry'] = \
            neighbors.loc[neighbors['median'].idxmax(), 'geometry']

    # we take those records where the max_gradient is in the upper 5% of all
    # gradients found
    field_boundaries.dropna(inplace=True, subset=['max_gradient'])
    max_gradient_q95 = field_boundaries['max_gradient'].quantile(0.95)
    field_boundaries = field_boundaries[
        field_boundaries['max_gradient'] >= max_gradient_q95].copy()

    # remove records, in which reference and neighbor geometry are flipped
    # These are strictly speaking duplicated records
    field_boundaries['dissolved_geoms'] = field_boundaries.apply(
        lambda x: x.geometry.union(x.neigbor_geometry).bounds, axis=1)
    field_boundaries.drop_duplicates(subset=['dissolved_geoms'], inplace=True)

    sel_cols = [
        'geometry', 'neigbor_geometry', 'min', 'max', 'median', 'max_gradient',
        'percentile_10', 'percentile_90', 'dissolved_geoms']

    return field_boundaries[sel_cols]

def plot_clipped_image(
    band: Band,
    parcel: pd.Series,
    iteration: int,
    f: plt.Figure = None
) -> plt.Figure:
    """
    Plot the clipped image and the reference and neighbor geometry.

    :param band: Band object.
    :param parcel: GeoSeries with the reference and neighbor geometry.
    :param round: plotting iteration (either 1 or 2).
    :param f: Figure object. Must be passed in iteration 2.
    :return: Figure object.
    """
    if iteration == 1:
        f, ax = plt.subplots(figsize=(20, 10), ncols=2)
        band.plot(ax=ax[0], colormap='viridis')
        # plot the reference geometry
        ref_bounds = parcel.geometry.exterior.xy
        ax[0].plot(*ref_bounds, color='black', linewidth=2,
            linestyle='dashed', label='reference')
        # plot the neighbor geometry
        neighbor_bounds = parcel.neigbor_geometry.exterior.xy
        ax[0].plot(*neighbor_bounds, color='black', linewidth=2,
            label='neighbor', linestyle='dotted')
        # add scalebar
        scalebar = ScaleBar(dx=1)
        ax[0].add_artist(scalebar)
        return f
    elif iteration == 2:
        ax = f.axes
        # plot the clipped image
        band.plot(ax=ax[1], colormap='viridis')
        ax[0].set_title('GLAI candidate parcels')
        ax[1].set_title('Zoom-in on GLAI gradient')
        ax[0].plot(
            *band.bounds.exterior.xy,
            color='red', linewidth=2, label='Zoom-in')
        # add legend
        ax[0].legend(loc='lower left')
        return f
    else:
        raise ValueError('iteration must be either 1 or 2.')

def extract_gradients_at_boundaries(
    field_bounaries_path: Path,
    sat_lai_path: Path,
    output_dir: Path,
    plot: bool = True
) -> None:
    """
    Extract the GLAI values at field boundaries that have a sharp gradient in
    median GLAI values.

    :param field_bounaries_path:
        Path to the field boundaries. These boundaries are used to identify
        field boundaries where there is a sharp gradient in median GLAI values.
        The boundaries themselves are not used for validation.
    :param sat_lai_path:
        Path to the GeoTiff image with the GLAI values.
    :param output_dir:
        Path to the output directory.
    :param plot:
        If True, plot the clipped image and the reference and neighbor geometry.
        Set to False to speed up the process.
    """
    # identify field boundaries where there is a sharp gradient in median GLAI
    # values
    boundaries_with_gradient = identify_boundaries_with_gradient(field_bounaries_path)

    # read data
    sat_lai = Band.from_rasterio(
        fpath_raster=sat_lai_path,
        band_idx=1,
        band_name_dst='lai'
    )

    # loop over the identified field parcels and compute the boundary from the imagery
    counter = 0
    for idx, row in boundaries_with_gradient.iterrows():

        # ignore MultiPolygons
        if row.geometry.type == 'MultiPolygon':
            print(f'Parcel {idx} is a MultiPolygon. Skipping.')
            counter += 1
            continue
        if row.neigbor_geometry.type == 'MultiPolygon':
            print(f'Parcel {idx} has a neighbor that is a MultiPolygon. Skipping.')
            counter += 1
            continue

        try:
            sat_lai_clipped = sat_lai.clip(
                clipping_bounds=row['dissolved_geoms']
            )
        except ValueError as e:
            # if the clipping bounds are overlapping the raster, continue
            print(f'Parcel {idx}: {e}')
            continue

        # Optional:
        # plot the clipped image and the reference and neighbor geometry
        f = plot_clipped_image(band=sat_lai_clipped, parcel=row, iteration=1)

        # get the actual intersection of the two geometries (reference and neighbor)
        geom_buffered = buffer_geom(row.geometry)
        intersection = geom_buffered.intersection(row.neigbor_geometry)

        # get the "long" side of the intersection since this is the side where
        # the gradient is actually to be considered
        intersection_bounds = intersection.bounds
        long_axis = 'x' if intersection_bounds[2] - intersection_bounds[0] > \
            intersection_bounds[3] - intersection_bounds[1] else 'y'

        # clip the image to the intersection
        sat_lai_clipped = sat_lai.clip(
            clipping_bounds=intersection.bounds
        )
        f = plot_clipped_image(
            band=sat_lai_clipped, parcel=row, iteration=2, f=f)

        # save the plot
        f.savefig(
            output_dir.joinpath(f'{idx}_reference_neighbor.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(f)

        # save the clipped image
        sat_lai_clipped.to_rasterio(
            fpath_raster=output_dir.joinpath(f'{idx}_reference_neighbor.tif'),
            band_name='lai'
        )

        print(
            f'Processed parcel {idx} ({counter}/{boundaries_with_gradient.shape[0]}).')
        counter += 1

def create_boundaries(name: str) -> None:
    """
    Main function of the script.

    :param data_dir: Path to the data directory.
    """

    parcel_stats_dir = VALIDATE_DIR.joinpath("ps_parcel_stats")
    data_dir = VALIDATE_DIR.joinpath(name)

    for month in data_dir.iterdir():
        for index in month.iterdir():
            if not index.name[:4].isdigit():
                continue
            print("---- INDEX: ", index.name)
            
            field_boundaries_path = parcel_stats_dir.joinpath(index.name[:4] + '_lai_4bands_parcel_stats.gpkg')
            out_dir = data_dir.joinpath(month.name, index.name[:4] + "_field_boundaries")
            out_dir.mkdir(exist_ok=True, parents=True)

            extract_gradients_at_boundaries(
                field_bounaries_path=field_boundaries_path,
                sat_lai_path=index,
                output_dir=out_dir
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bands", type=Path, help="Select bands. Either '4b' or '8b'", required=True)
    args = parser.parse_args()
    create_boundaries(args.bands)