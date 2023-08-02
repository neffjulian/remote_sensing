from pathlib import Path

from eodal.core.raster import RasterCollection

def compare_field_boundaries(raster_path: Path, field_boundary_path: Path) -> None:
    raster = RasterCollection(raster_path)
    field_boundary = RasterCollection(field_boundary_path)

    