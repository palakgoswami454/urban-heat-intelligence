"""Download and align satellite data for the study area from Microsoft Planetary Computer.

For every clear pre-monsoon Landsat 8/9 scene we build one aligned 30 m "scene":
  * Land Surface Temperature (LST, deg C) from Landsat Collection-2 Level-2 band ST_B10
  * a Sentinel-2 L2A cloud-masked median composite (+/- N days) of 6 bands
  * spectral indices NDVI, NDBI, MNDWI
  * Copernicus 30 m DEM (elevation)
Each scene is saved as a compressed .npz file. No account or API key is needed.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import warnings
from collections import defaultdict

import numpy as np

from .config import CHANNELS, Config
from .grid import make_grid, save_grid

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
S2_BAD_SCL = [0, 1, 3, 8, 9, 10, 11]  # nodata, saturated, shadow, cloud med/high, cirrus, snow
LANDSAT_BAD_BITS = (1, 2, 3, 4)  # dilated cloud, cirrus, cloud, cloud shadow
S2_OFFSET_DATE = dt.date(2022, 1, 25)  # processing baseline 04.00 adds a -1000 DN offset

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _client():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _geobox(grid: dict):
    from affine import Affine
    from odc.geo.geobox import GeoBox

    r = grid["resolution"]
    aff = Affine(r, 0, grid["origin_x"], 0, -r, grid["origin_y"])
    return GeoBox((grid["height"], grid["width"]), aff, grid["crs"])


def _retry(fn, tries=4, wait=10):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network hiccups are common on long downloads
            if i == tries - 1:
                raise
            print(f"   retrying after error: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(wait * (i + 1))


def _search(client, collection, bbox, start, end, query=None):
    s = client.search(collections=[collection], bbox=bbox,
                      datetime=f"{start}/{end}", query=query or {})
    return list(s.items())


def find_landsat_dates(cfg: Config, client) -> list[tuple[str, list]]:
    """Pick the clearest summer Landsat 8/9 acquisition dates per year."""
    chosen = []
    for year in cfg.years:
        items = _retry(lambda: _search(
            client, "landsat-c2-l2", cfg.bbox,
            f"{year}-{cfg.season[0]}", f"{year}-{cfg.season[1]}",
            {"eo:cloud_cover": {"lt": cfg.landsat_max_cloud},
             "platform": {"in": ["landsat-8", "landsat-9"]}}))
        by_date = defaultdict(list)
        for it in items:
            by_date[it.datetime.date().isoformat()].append(it)
        ranked = sorted(by_date.items(),
                        key=lambda kv: np.mean([i.properties.get("eo:cloud_cover", 100) for i in kv[1]]))
        picks = []
        for d, its in ranked:  # spread picks over the season (>= 20 days apart)
            if all(abs((dt.date.fromisoformat(d) - dt.date.fromisoformat(p)).days) >= 20 for p, _ in picks):
                picks.append((d, its))
            if len(picks) >= cfg.max_scenes_per_year:
                break
        print(f"  {year}: {len(by_date)} candidate dates -> using {[p for p, _ in picks]}")
        chosen.extend(picks)
    chosen.sort(key=lambda kv: kv[0])
    if len(chosen) > cfg.max_scenes:  # keep an even spread over the years
        idx = np.linspace(0, len(chosen) - 1, cfg.max_scenes).round().astype(int)
        chosen = [chosen[i] for i in sorted(set(idx))]
    return chosen


def load_landsat_lst(items, gbox):
    import odc.stac

    ds = _retry(lambda: odc.stac.load(
        items, bands=["lwir11", "qa_pixel"], geobox=gbox, groupby="solar_day",
        resampling={"lwir11": "bilinear", "qa_pixel": "nearest"}, pool=4))
    st = ds["lwir11"].isel(time=0).values.astype("float32")
    qa = ds["qa_pixel"].isel(time=0).values.astype("uint16")
    lst = st * 0.00341802 + 149.0 - 273.15
    bad = (st == 0) | (qa & 1).astype(bool)
    for b in LANDSAT_BAD_BITS:
        bad |= ((qa >> b) & 1).astype(bool)
    lst[bad | (lst < 0) | (lst > 80)] = np.nan
    return lst


def load_s2_composite(client, cfg: Config, date: str, gbox):
    import odc.stac

    d = dt.date.fromisoformat(date)
    start = (d - dt.timedelta(days=cfg.s2_window_days)).isoformat()
    end = (d + dt.timedelta(days=cfg.s2_window_days)).isoformat()
    items = _retry(lambda: _search(client, "sentinel-2-l2a", cfg.bbox, start, end,
                                   {"eo:cloud_cover": {"lt": cfg.s2_max_cloud}}))
    if not items:
        return None
    ds = _retry(lambda: odc.stac.load(
        items, bands=S2_BANDS + ["SCL"], geobox=gbox, groupby="solar_day",
        resampling={**{b: "average" for b in S2_BANDS}, "SCL": "nearest"}, pool=4))
    stacks = []
    for t in range(ds.sizes["time"]):
        tdate = dt.date.fromisoformat(str(ds.time.values[t])[:10])
        offset = 1000.0 if tdate >= S2_OFFSET_DATE else 0.0
        scl = ds["SCL"].isel(time=t).values
        bad = np.isin(scl, S2_BAD_SCL)
        bands = []
        for b in S2_BANDS:
            dn = ds[b].isel(time=t).values.astype("float32")
            refl = (dn - offset) / 10000.0
            refl[(dn == 0) | bad] = np.nan
            bands.append(np.clip(refl, 0, 1.5))
        stacks.append(np.stack(bands))
    return np.nanmedian(np.stack(stacks), axis=0)  # (6, H, W)


def load_dem(client, cfg: Config, gbox):
    import odc.stac

    items = _retry(lambda: _search(client, "cop-dem-glo-30", cfg.bbox, "2000-01-01", "2030-01-01"))
    ds = _retry(lambda: odc.stac.load(items, bands=["data"], geobox=gbox,
                                      resampling="bilinear", pool=4))
    dem = ds["data"].max(dim="time").values.astype("float32")
    dem[dem < -100] = np.nan
    return dem


# ESA WorldCover 2021 (10 m) classes, aggregated to fractions of each 30 m pixel
WORLDCOVER_CLASSES = {"tree": [10], "shrub_grass": [20, 30], "cropland": [40], "built": [50],
                      "bare": [60], "water": [80, 90, 95], "other": [70, 100]}


def worldcover_fractions(grid: dict, client) -> dict:
    """Land-cover fractions per 30 m pixel from ESA WorldCover 2021 (Planetary Computer)."""
    import odc.stac
    from affine import Affine
    from odc.geo.geobox import GeoBox

    f = 3  # 10 m sub-pixels per 30 m pixel side
    r = grid["resolution"] / f
    gb = GeoBox((grid["height"] * f, grid["width"] * f), Affine(r, 0, grid["origin_x"], 0, -r, grid["origin_y"]),
                grid["crs"])
    lon0, lat0, lon1, lat1 = _grid_bbox(grid)
    items = _retry(lambda: _search(client, "esa-worldcover", [lon0, lat0, lon1, lat1], "2021-01-01", "2021-12-31"))
    if not items:
        raise RuntimeError("no WorldCover items found")
    ds = _retry(lambda: odc.stac.load(items, bands=["map"], geobox=gb, resampling="nearest", pool=4))
    lc = ds["map"].max(dim="time").values
    H, W = grid["height"], grid["width"]
    out = {}
    for name, codes in WORLDCOVER_CLASSES.items():
        m = np.isin(lc, codes).astype("float32")
        out[name] = m.reshape(H, f, W, f).mean(axis=(1, 3)).astype("float16")
    return out


def _grid_bbox(grid):
    from .grid import pixel_to_lonlat

    lon, lat = pixel_to_lonlat(grid, [0, 0, grid["height"], grid["height"]], [0, grid["width"], grid["width"], 0])
    return float(min(lon)), float(min(lat)), float(max(lon)), float(max(lat))


def worldcover_path(cfg: Config):
    return cfg.data_dir / ("worldcover_demo.npz" if cfg.demo else "worldcover.npz")


def download_worldcover(cfg: Config, grid: dict, client) -> None:
    path = worldcover_path(cfg)
    if path.exists():
        return
    try:
        print("Loading ESA WorldCover 2021 land cover ...")
        fr = worldcover_fractions(grid, client)
        np.savez_compressed(path, **fr)
        print("   built-up {:.0%}, cropland {:.0%}, bare {:.0%}, tree {:.0%} of study area".format(
            *(float(fr[k].astype("float32").mean()) for k in ("built", "cropland", "bare", "tree"))))
    except Exception as e:  # analysis falls back to spectral indices
        print(f"   WorldCover unavailable ({type(e).__name__}: {e}); using spectral-index land cover instead")


def spectral_indices(s2: np.ndarray) -> np.ndarray:
    b03, b04, b08, b11 = s2[1], s2[2], s2[3], s2[4]
    eps = 1e-6
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndbi = (b11 - b08) / (b11 + b08 + eps)
    mndwi = (b03 - b11) / (b03 + b11 + eps)
    return np.stack([np.clip(ndvi, -1, 1), np.clip(ndbi, -1, 1), np.clip(mndwi, -1, 1)])


def save_scene(path, date, X, lst, source):
    np.savez_compressed(path, X=X.astype("float16"), lst=lst.astype("float16"),
                        date=date, source=source, channels=np.array(CHANNELS))


def download_scenes(cfg: Config) -> list[str]:
    """Download all scenes for the study area. Skips scenes already on disk."""
    grid = make_grid(cfg.bbox, cfg.utm_crs, cfg.resolution)
    save_grid(grid, cfg.grid_path)
    gbox = _geobox(grid)
    print(f"Grid {grid['height']}x{grid['width']} px at {cfg.resolution} m ({cfg.utm_crs})")

    client = _client()
    dem_path = cfg.data_dir / "dem.npy"
    if dem_path.exists():
        dem = np.load(dem_path)
    else:
        print("Loading Copernicus DEM ...")
        dem = load_dem(client, cfg, gbox)
        np.save(dem_path, dem)
    download_worldcover(cfg, grid, client)

    print("Searching Landsat 8/9 summer scenes ...")
    dates = find_landsat_dates(cfg, client)
    saved = []
    log = []
    for date, items in dates:
        out = cfg.scenes_dir / f"scene_{date}.npz"
        if out.exists():
            saved.append(out.name)
            continue
        print(f"Scene {date}: Landsat LST ...")
        lst = load_landsat_lst(items, gbox)
        print(f"Scene {date}: Sentinel-2 composite ...")
        s2 = load_s2_composite(client, cfg, date, gbox)
        if s2 is None:
            print("   no Sentinel-2 data, skipped")
            continue
        X = np.concatenate([s2, spectral_indices(s2), dem[None]], axis=0)
        valid = np.isfinite(lst) & np.isfinite(X).all(axis=0)
        frac = float(valid.mean())
        log.append({"date": date, "valid_fraction": round(frac, 3),
                    "landsat_items": [i.id for i in items]})
        if frac < cfg.min_valid_fraction:
            print(f"   only {frac:.0%} valid pixels, skipped")
            continue
        save_scene(out, date, X, lst, source="planetary-computer")
        saved.append(out.name)
        print(f"   saved ({frac:.0%} valid, median LST {np.nanmedian(lst):.1f} C)")
    (cfg.data_dir / "download_log.json").write_text(json.dumps(log, indent=2))
    print(f"{len(saved)} scenes ready in {cfg.scenes_dir}")
    if len(saved) < 3:
        raise RuntimeError("Fewer than 3 usable scenes - relax cloud thresholds in config.py")
    return saved
