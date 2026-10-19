"""Daily weather history for the city from the free Open-Meteo archive (ERA5 reanalysis)."""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd

from .config import Config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Same variable names exist in the Open-Meteo forecast API, which the web dashboard
# uses to run the trained model live in the browser.
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
]


def download_weather(cfg: Config) -> pd.DataFrame:
    import requests

    path = cfg.data_dir / "weather_daily.csv"
    end = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    params = {
        "latitude": cfg.point[0],
        "longitude": cfg.point[1],
        "start_date": cfg.weather_start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(4):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=120)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retrying Open-Meteo ({e})")
            time.sleep(15 * (attempt + 1))
    daily = r.json()["daily"]
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").astype(float).interpolate(limit=5).dropna()
    df.to_csv(path)
    print(f"Weather: {len(df)} days {df.index.min().date()} .. {df.index.max().date()}")
    return df


def synthetic_weather(cfg: Config) -> pd.DataFrame:
    """Plausible synthetic daily weather (DEMO / testing only)."""
    rng = np.random.default_rng(cfg.seed)
    idx = pd.date_range(cfg.weather_start, "2025-12-31", freq="D")
    doy = idx.dayofyear.values
    season = 31.5 + 6.5 * np.sin(2 * np.pi * (doy - 30) / 365) - 3.5 * np.exp(-((doy - 200) / 40) ** 2)
    trend = 0.03 * (idx.year.values - 2000)
    noise = np.zeros(len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.75 * noise[i - 1] + rng.normal(0, 0.9)
    tmax = season + trend + noise
    tmin = tmax - 11 + 3 * np.exp(-((doy - 200) / 50) ** 2) + rng.normal(0, 0.8, len(idx))
    rad = 20 + 6 * np.sin(2 * np.pi * (doy - 30) / 365) - 7 * np.exp(-((doy - 200) / 45) ** 2) + rng.normal(0, 1.5, len(idx))
    rain = np.where(rng.random(len(idx)) < 0.6 * np.exp(-((doy - 200) / 45) ** 2), rng.gamma(2, 6, len(idx)), 0)
    df = pd.DataFrame({
        "temperature_2m_max": tmax,
        "temperature_2m_min": tmin,
        "temperature_2m_mean": (tmax + tmin) / 2,
        "shortwave_radiation_sum": rad,
        "wind_speed_10m_max": 12 + 6 * np.exp(-((doy - 190) / 40) ** 2) + rng.normal(0, 2, len(idx)),
        "precipitation_sum": rain,
        "et0_fao_evapotranspiration": 3 + 0.12 * (tmax - 25) + rng.normal(0, 0.4, len(idx)),
    }, index=idx)
    df.index.name = "date"
    df.to_csv(cfg.data_dir / "weather_daily_demo.csv")
    return df
