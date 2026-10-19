"""Central configuration for the Urban Heat Intelligence pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ---- Study area: Pune Municipal Corporation + Pimpri-Chinchwad (lon/lat) ----
    city: str = "Pune"
    bbox: tuple = (73.72, 18.42, 74.02, 18.68)  # min_lon, min_lat, max_lon, max_lat
    utm_crs: str = "EPSG:32643"  # UTM zone 43N
    resolution: int = 30  # metres (Landsat thermal grid)

    # ---- Satellite sampling: pre-monsoon summer (peak urban heat) ----
    years: tuple = (2019, 2020, 2021, 2022, 2023, 2024, 2025)
    season: tuple = ("03-01", "05-31")
    landsat_max_cloud: float = 20.0
    s2_max_cloud: float = 30.0
    s2_window_days: int = 8  # Sentinel-2 composite window around each Landsat date
    max_scenes: int = 18
    max_scenes_per_year: int = 3
    min_valid_fraction: float = 0.6

    # ---- Deep learning (LST model) ----
    patch: int = 64
    train_stride: int = 48
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    base_channels: int = 32
    patience: int = 6
    seed: int = 42
    n_blocks: int = 4  # spatial cross-validation blocks per axis (4x4 = 16)

    # ---- Grid used for hotspot / recommendation analysis ----
    cell_px: int = 17  # 17 px x 30 m ~ 510 m cells

    # ---- Weather forecasting (LSTM) ----
    point: tuple = (18.52, 73.86)  # Pune city centre (lat, lon)
    weather_start: str = "2000-01-01"
    window: int = 30
    horizon: int = 7
    lstm_hidden: int = 64
    fc_epochs: int = 40
    fc_train_end: str = "2019-12-31"
    fc_val_end: str = "2022-12-31"

    demo: bool = False

    # ---- Paths ----
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    out_dir: Path = field(default_factory=lambda: ROOT / "outputs")
    web_data_dir: Path = field(default_factory=lambda: ROOT / "web" / "data")

    @property
    def scenes_dir(self) -> Path:
        return self.data_dir / ("scenes_demo" if self.demo else "scenes")

    @property
    def grid_path(self) -> Path:
        return self.data_dir / ("grid_demo.json" if self.demo else "grid.json")

    @property
    def fig_dir(self) -> Path:
        return self.out_dir / "figures"

    @property
    def model_dir(self) -> Path:
        return self.out_dir / "models"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.scenes_dir, self.out_dir, self.fig_dir,
                  self.model_dir, self.web_data_dir):
            p.mkdir(parents=True, exist_ok=True)


# Input channels of the LST model and their explainability groups
CHANNELS = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDBI", "MNDWI", "DEM"]
GROUPS = {
    "Vegetation": ["B08", "NDVI"],
    "Built-up / impervious": ["B11", "B12", "NDBI"],
    "Water / moisture": ["MNDWI"],
    "Terrain (elevation)": ["DEM"],
    "Visible reflectance (albedo)": ["B02", "B03", "B04"],
}


def get_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    if os.environ.get("UHI_DEMO") == "1":
        cfg.demo = True
    cfg.ensure_dirs()
    return cfg
