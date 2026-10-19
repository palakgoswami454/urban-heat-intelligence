"""Study-area raster grid (UTM) and conversions to longitude/latitude."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from pyproj import Transformer


def make_grid(bbox, crs: str, resolution: float) -> dict:
    """Build a north-up UTM grid that covers a lon/lat bounding box."""
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lons = [bbox[0], bbox[2], bbox[0], bbox[2]]
    lats = [bbox[1], bbox[1], bbox[3], bbox[3]]
    xs, ys = to_utm.transform(lons, lats)
    minx = math.floor(min(xs) / resolution) * resolution
    maxx = math.ceil(max(xs) / resolution) * resolution
    miny = math.floor(min(ys) / resolution) * resolution
    maxy = math.ceil(max(ys) / resolution) * resolution
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    return {
        "crs": crs,
        "resolution": resolution,
        "origin_x": minx,  # left edge
        "origin_y": maxy,  # top edge
        "height": height,
        "width": width,
    }


def save_grid(grid: dict, path: Path) -> None:
    path.write_text(json.dumps(grid, indent=2))


def load_grid(path: Path) -> dict:
    return json.loads(path.read_text())


def pixel_to_lonlat(grid: dict, rows, cols):
    """Convert (fractional) pixel coordinates to lon/lat. rows/cols may be arrays."""
    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols, dtype=float)
    x = grid["origin_x"] + cols * grid["resolution"]
    y = grid["origin_y"] - rows * grid["resolution"]
    to_ll = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(x, y)
    return np.asarray(lon), np.asarray(lat)
