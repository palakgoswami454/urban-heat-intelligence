"""Export model outputs as static files for the web dashboard (web/data)."""
from __future__ import annotations

import datetime as dt
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import RECS
from .config import GROUPS, Config
from .grid import load_grid, pixel_to_lonlat


def _overlay(a, path, cmap, vmin, vmax):
    norm = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    rgba = plt.get_cmap(cmap)(np.nan_to_num(norm))
    rgba[..., 3] = np.where(np.isfinite(a), 0.78, 0)
    plt.imsave(path, rgba)


def run(cfg: Config):
    grid = load_grid(cfg.grid_path)
    cells = json.loads((cfg.out_dir / "cells.json").read_text())
    summ = json.loads((cfg.out_dir / "analysis_summary.json").read_text())
    mlst = json.loads((cfg.out_dir / "metrics_lst.json").read_text())
    fpath = cfg.out_dir / "metrics_forecast.json"
    mfc = json.loads(fpath.read_text()) if fpath.exists() else None
    maps = np.load(cfg.out_dir / "analysis_maps.npz")

    H, W = grid["height"], grid["width"]
    lon, lat = pixel_to_lonlat(grid, [0, 0, H, H], [0, W, W, 0])
    bounds = [[float(min(lat)), float(min(lon))], [float(max(lat)), float(max(lon))]]
    d = cfg.web_data_dir
    hv = float(np.nanpercentile(np.abs(maps["heat"]), 98))
    _overlay(maps["heat"], d / "heat.png", "RdYlBu_r", -hv, hv)
    lo, hi = np.nanpercentile(maps["lst"], [2, 98])
    _overlay(maps["lst"], d / "lst.png", "inferno", lo, hi)
    cool = np.minimum(maps["cool_green"], maps["cool_roof"])
    cool[cool > -0.05] = np.nan
    cv = float(np.nanpercentile(-cool, 98)) if np.isfinite(cool).any() else 1.0
    _overlay(-cool, d / "cooling.png", "Blues", 0, cv)

    meta = {
        "city": cfg.city, "demo": cfg.demo,
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "reference_scene": summ["reference_scene"], "reference_median_lst_c": summ["reference_median_lst_c"],
        "years": summ["years"], "recent_years": summ["recent_years"],
        "hotspot_threshold_c": summ["hotspot_threshold_c"], "cell_size_m": summ["cell_size_m"],
        "n_cells": summ["n_cells"], "n_hotspots": summ["n_hotspots"], "bounds": bounds,
        "overlay_ranges": {"heat": [-hv, hv], "lst": [float(lo), float(hi)], "cooling": [0, cv]},
        "recs": RECS, "groups": list(GROUPS), "suhi_by_year": summ["suhi_by_year"],
        "recommendation_counts": summ["recommendation_counts"],
        "n_urban_hotspots": summ.get("n_urban_hotspots"), "land_cover_source": summ.get("land_cover_source"),
        "urban_hotspot_recommendations": summ.get("urban_hotspot_recommendations"),
        "heat_by_land_cover": summ.get("heat_by_land_cover"),
        "mean_heat_urban_cells_c": summ.get("mean_heat_urban_cells_c"),
        "mean_heat_fallow_cells_c": summ.get("mean_heat_fallow_cells_c"),
    }
    (d / "grid.json").write_text(json.dumps({"meta": meta, "cells": cells}, separators=(",", ":")))

    best = mlst["best_model"]
    metrics = {
        "demo": cfg.demo,
        "lst": {"best_model": best, "results": mlst["results"], "n_scenes": mlst["n_scenes"],
                "scenes": mlst["scenes"], "patches": mlst["patches"]},
        "importance": summ["permutation_importance_rmse_increase"],
        "cooling": {"greening_hotspots_c": summ["mean_cooling_greening_hotspots_c"],
                    "roofs_hotspots_c": summ["mean_cooling_roofs_hotspots_c"],
                    "greening_urban_hotspots_c": summ.get("mean_cooling_greening_urban_hotspots_c"),
                    "roofs_urban_hotspots_c": summ.get("mean_cooling_roofs_urban_hotspots_c")},
    }
    if mfc:
        metrics["forecast"] = {k: mfc[k] for k in ("metrics", "heat_stress_threshold_c", "threshold_definition",
                                                    "train_period", "test_period", "attention_by_lag")}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_results_md(cfg, mlst, summ, mfc)
    print(f"Web data exported to {d}")


def write_results_md(cfg, mlst, summ, mfc):
    """Human-readable results summary with the real numbers (outputs/RESULTS.md)."""
    L = ["# Results", ""]
    if cfg.demo:
        L += ["> **DEMO RUN - synthetic data. These numbers are NOT real results.**", ""]
    L += [f"Generated {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC. Study area: {cfg.city}. "
          f"{mlst['n_scenes']} summer scenes from {mlst['scenes'][0]} to {mlst['scenes'][-1]}.", "",
          "## 1. Land surface temperature model (test = held-out spatial blocks)", "",
          "| Model | RMSE (C) | MAE (C) | R2 | Hotspot F1 | Params |", "|---|---|---|---|---|---|"]
    for k, r in mlst["results"].items():
        star = " **(selected)**" if k == mlst["best_model"] else ""
        L.append(f"| {k}{star} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.3f} | {r['hotspot_f1']:.3f} | "
                 f"{r.get('params', '-')} |")
    L += ["", f"Model selection: {mlst['selection']}. Patches: {mlst['patches']}.", "",
          "## 2. Explainability - permutation importance (increase in test RMSE, C)", ""]
    for g, v in sorted(summ["permutation_importance_rmse_increase"].items(), key=lambda kv: -kv[1]):
        L.append(f"- {g}: {v:.3f}")
    L += ["", "## 3. Hotspots and cooling scenarios", "",
          f"- Reference scene: {summ['reference_scene']} (city median LST {summ['reference_median_lst_c']} C)",
          f"- Grid cells: {summ['n_cells']} of ~{summ['cell_size_m']} m; hotspots: {summ['n_hotspots']} "
          f"(heat anomaly >= {summ['hotspot_threshold_c']} C, years {summ['recent_years']})",
          f"- Mean modelled surface cooling in hotspots: greening {summ['mean_cooling_greening_hotspots_c']:.2f} C, "
          f"cool roofs {summ['mean_cooling_roofs_hotspots_c']:.2f} C",
          f"- Recommendations for hotspots: {summ['recommendation_counts']}",
          f"- Land cover source: {summ.get('land_cover_source')}",
          f"- Hotspots by dominant cover: {summ.get('hotspot_dominant_cover')}",
          f"- Hotspots in built-up areas (urban hotspots): {summ.get('n_urban_hotspots')}; their recommendations: "
          f"{summ.get('urban_hotspot_recommendations')}",
          f"- Mean modelled surface cooling in urban hotspots: greening "
          f"{summ.get('mean_cooling_greening_urban_hotspots_c', 0):.2f} C, cool roofs "
          f"{summ.get('mean_cooling_roofs_urban_hotspots_c', 0):.2f} C",
          f"- Mean heat anomaly of mostly built-up cells: {summ.get('mean_heat_urban_cells_c')} C; "
          f"of mostly bare/fallow cells: {summ.get('mean_heat_fallow_cells_c')} C", "",
          "Heat anomaly by land cover (recent summers, pixels with >=50% of the class):", ""]
    for k, v in (summ.get("heat_by_land_cover") or {}).items():
        L.append(f"- {k}: mean {v['mean']:+.2f} C, median {v['q'][2]:+.2f} C (n={v['n_pixels']})")
    L += ["", "Surface heat contrast per year (C):", "", "| Year | Built-up minus vegetated | Built-up minus bare/fallow |",
          "|---|---|---|"]
    for s in summ["suhi_by_year"]:
        f = lambda v: "-" if v is None else f"{v:+.2f}"
        L.append(f"| {s['year']} | {f(s['suhi_c'])} | {f(s.get('urban_minus_fallow_c'))} |")
    if mfc:
        L += ["", "## 4. 7-day max temperature forecast (test period " + mfc["test_period"] + ")", "",
              "| Method | MAE all leads | MAE +1d | MAE +3d | MAE +7d | Heat-day F1 +1d | Heat-day F1 +3d |",
              "|---|---|---|---|---|---|---|"]
        for k, m in mfc["metrics"].items():
            L.append(f"| {k} | {m['mae']:.3f} | {m['mae_by_lead'][0]:.3f} | {m['mae_by_lead'][2]:.3f} | "
                     f"{m['mae_by_lead'][-1]:.3f} | {m['heat_day']['lead_1']['f1']:.3f} | {m['heat_day']['lead_3']['f1']:.3f} |")
        L += ["", f"Heat-stress day = daily max >= {mfc['heat_stress_threshold_c']} C ({mfc['threshold_definition']})."]
    L += ["", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in sorted(cfg.fig_dir.glob("*.png"))]
    (cfg.out_dir / "RESULTS.md").write_text("\n".join(L) + "\n")
