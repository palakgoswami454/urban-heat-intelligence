"""Synthetic city scenes for DEMO mode and automated testing.

They mimic the structure of the real data (same channels, same file format, same
grid over Pune) so the whole pipeline and the dashboard can be exercised without
internet access. Results produced in demo mode are NOT real and are labelled as such.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .data_satellite import save_scene, spectral_indices
from .grid import make_grid, save_grid


def _smooth(a, k=9):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(a, k)


def make_demo_scenes(cfg: Config, n_scenes: int = 10, resolution: int = 80) -> list[str]:
    rng = np.random.default_rng(cfg.seed)
    grid = make_grid(cfg.bbox, cfg.utm_crs, resolution)
    save_grid(grid, cfg.grid_path)
    H, W = grid["height"], grid["width"]
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)

    # static landscape: hills, river, city core, satellite towns that grow over time
    hills = np.clip(_smooth(rng.normal(0, 1, (H, W)), 25) * 40, 0, None)
    hills += 250 * np.exp(-(((xx - 0.15) ** 2 + (yy - 0.85) ** 2) / 0.02))
    dem = 550 + hills + 20 * _smooth(rng.normal(0, 1, (H, W)), 5)
    river = np.exp(-((yy - 0.45 - 0.08 * np.sin(xx * 9)) ** 2) / 0.00025)
    core = np.exp(-(((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.03))
    towns = [(0.75, 0.25), (0.25, 0.3), (0.8, 0.7), (0.4, 0.8), (0.62, 0.38)]
    texture = _smooth(rng.normal(0, 1, (H, W)), 3)
    parks = (_smooth(rng.normal(0, 1, (H, W)), 6) > 0.12).astype(float)

    years = list(cfg.years)
    saved = []
    for s in range(n_scenes):
        year = years[int(s * len(years) / n_scenes)]
        growth = (year - years[0]) / max(1, years[-1] - years[0])
        built = 0.9 * core
        for i, (tx, ty) in enumerate(towns):
            r2 = 0.004 + 0.012 * growth * (1 + i % 2)
            built += 0.8 * np.exp(-(((xx - tx) ** 2 + (yy - ty) ** 2) / r2))
        built = np.clip(built + 0.15 * texture - 0.5 * parks * core, 0, 1)
        veg = np.clip(0.55 - 0.6 * built + 0.25 * hills / 300 + 0.3 * parks + 0.1 * texture, 0.02, 0.95)
        water = np.clip(river * (1 - built), 0, 1)

        b02 = 0.06 + 0.07 * built - 0.02 * veg
        b03 = 0.08 + 0.07 * built + 0.01 * veg - 0.02 * water
        b04 = 0.09 + 0.10 * built - 0.06 * veg - 0.04 * water
        b08 = 0.18 + 0.30 * veg + 0.03 * built - 0.15 * water
        b11 = 0.22 + 0.10 * built - 0.06 * veg - 0.18 * water
        b12 = 0.17 + 0.12 * built - 0.08 * veg - 0.15 * water
        s2 = np.stack([b02, b03, b04, b08, b11, b12]) + rng.normal(0, 0.005, (6, H, W))
        s2 = np.clip(s2, 0.001, 1)
        idx = spectral_indices(s2)
        X = np.concatenate([s2, idx, dem[None]], axis=0).astype("float32")

        base = 38 + 4 * rng.random() + 0.3 * (year - years[0])
        lst = (base + 7.5 * built - 6.0 * veg - 9.0 * water - 0.012 * (dem - 550)
               + 2.0 * _smooth(built, 4) + 0.8 * _smooth(rng.normal(0, 1, (H, W)), 4)
               + rng.normal(0, 0.4, (H, W)))
        clouds = _smooth(rng.normal(0, 1, (H, W)), 12) > 0.25
        lst[clouds] = np.nan
        date = f"{year}-{4 + s % 2:02d}-{10 + (s * 7) % 18:02d}"
        out = cfg.scenes_dir / f"scene_{date}.npz"
        save_scene(out, date, X, lst, source="SYNTHETIC-DEMO")
        saved.append(out.name)
    np.save(cfg.data_dir / "dem_demo.npy", dem.astype("float32"))
    b = np.clip(built, 0, 1); w = np.clip(water, 0, 1) * (1 - b)
    t = np.clip(veg - 0.45, 0, 1) * 1.5 * (1 - b - w)
    rest = np.clip(1 - b - w - t, 0, 1)
    from .data_satellite import worldcover_path
    np.savez_compressed(worldcover_path(cfg), built=b.astype("float16"), water=w.astype("float16"),
                        tree=t.astype("float16"), shrub_grass=(0.2 * rest).astype("float16"),
                        cropland=(0.5 * rest).astype("float16"), bare=(0.3 * rest).astype("float16"),
                        other=np.zeros_like(b, dtype="float16"))
    print(f"[demo] {len(saved)} synthetic scenes written to {cfg.scenes_dir}")
    return saved
