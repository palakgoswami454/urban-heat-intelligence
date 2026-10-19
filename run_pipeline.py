#!/usr/bin/env python
"""End-to-end pipeline for Urban Heat Intelligence.

Examples
  python run_pipeline.py                      # full run on real satellite + weather data
  python run_pipeline.py --stage train        # one stage only
  python run_pipeline.py --demo               # synthetic data (offline smoke test, NOT real results)

Stages: download -> train -> analyze -> forecast -> figures -> export
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from uhi.config import get_config  # noqa: E402

STAGES = ["download", "train", "analyze", "forecast", "figures", "export"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (testing only)")
    ap.add_argument("--max-scenes", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--fc-epochs", type=int)
    ap.add_argument("--models", default="pixel_mlp,unet,attention_unet")
    a = ap.parse_args()

    overrides = {"max_scenes": a.max_scenes, "epochs": a.epochs, "fc_epochs": a.fc_epochs}
    if a.demo:
        overrides.update({"demo": True, "patch": 32, "train_stride": 16, "base_channels": 16,
                          "cell_px": 8, "weather_start": "2005-01-01"})
    cfg = get_config(**overrides)
    stages = STAGES if a.stage == "all" else [a.stage]
    print(f"Urban Heat Intelligence | city={cfg.city} | demo={cfg.demo} | stages={stages}")

    for st in stages:
        t0 = time.time()
        print(f"\n===== {st.upper()} =====")
        if st == "download":
            if cfg.demo:
                from uhi.synthetic import make_demo_scenes
                from uhi.data_weather import synthetic_weather
                make_demo_scenes(cfg)
                synthetic_weather(cfg)
            else:
                from uhi.data_satellite import download_scenes
                from uhi.data_weather import download_weather
                download_scenes(cfg)
                download_weather(cfg)
        elif st == "train":
            from uhi import train_lst
            train_lst.run(cfg, models=a.models.split(","))
        elif st == "analyze":
            from uhi import analysis
            analysis.run(cfg)
        elif st == "forecast":
            from uhi import forecast
            name = "weather_daily_demo.csv" if cfg.demo else "weather_daily.csv"
            df = pd.read_csv(cfg.data_dir / name, index_col="date", parse_dates=True)
            forecast.run(cfg, df)
        elif st == "figures":
            from uhi import figures
            figures.run(cfg)
        elif st == "export":
            from uhi import export_web
            export_web.run(cfg)
        print(f"----- {st} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
