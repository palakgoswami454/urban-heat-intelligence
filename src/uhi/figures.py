"""Publication-style figures for the report and README."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import GROUPS, Config
from .dataset import block_ids, load_scenes

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
HEAT = "RdYlBu_r"
NAMES = {"city_median (no model)": "City median\n(no model)", "linear_regression": "Linear\nregression",
         "gradient_boosting": "Gradient\nboosting", "pixel_mlp": "Pixel MLP\n(no context)",
         "unet": "U-Net", "attention_unet": "Attention\nU-Net"}


def _save(fig, cfg, name, demo):
    if demo:
        fig.text(0.5, 0.5, "DEMO DATA", fontsize=40, color="grey", alpha=0.25, ha="center", va="center",
                 rotation=25, transform=fig.transFigure)
    fig.savefig(cfg.fig_dir / f"{name}.png")
    plt.close(fig)


def _img(ax, a, title, cmap=HEAT, vmin=None, vmax=None, label=""):
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=label)


def run(cfg: Config):
    demo = cfg.demo
    mlst = json.loads((cfg.out_dir / "metrics_lst.json").read_text())
    hist = json.loads((cfg.out_dir / "history_lst.json").read_text())
    summ = json.loads((cfg.out_dir / "analysis_summary.json").read_text())
    maps = np.load(cfg.out_dir / "analysis_maps.npz")
    scenes = load_scenes(cfg)
    ref = next(s for s in scenes if s.date == summ["reference_scene"])
    split = json.loads((cfg.model_dir / "split.json").read_text())["split"]
    H, W = ref.lst.shape
    test_mask = np.isin(block_ids(H, W, cfg.n_blocks), split["test"])

    # 1 study area
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    rgb = np.nan_to_num(ref.X[[2, 1, 0]].transpose(1, 2, 0))
    ax[0].imshow(np.clip(rgb / np.nanpercentile(rgb, 98), 0, 1)); ax[0].set_title(f"Sentinel-2 true colour ({ref.date})")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    _img(ax[1], ref.lst, "Landsat land surface temperature", label="deg C")
    _img(ax[2], ref.X[9], "Elevation (Copernicus DEM)", cmap="terrain", label="m")
    _save(fig, cfg, "01_study_area", demo)

    # 2 training curves
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, h in hist.items():
        ax.plot([e["epoch"] for e in h], [e["val_rmse"] for e in h], label=NAMES.get(name, name).replace("\n", " "))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation RMSE (deg C)"); ax.legend(); ax.set_title("Training convergence")
    _save(fig, cfg, "02_training_curves", demo)

    # 3 model comparison
    res = mlst["results"]
    names = list(res)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    best = mlst["best_model"]
    cols = ["#d95f02" if n == best else "#7570b3" if n in ("pixel_mlp", "unet", "attention_unet") else "#999999" for n in names]
    for a, key, lab in zip(ax, ["rmse", "r2", "hotspot_f1"], ["RMSE (deg C, lower better)", "R2 (higher better)",
                                                               "Hotspot F1 (higher better)"]):
        vals = [res[n][key] for n in names]
        a.bar(range(len(names)), vals, color=cols)
        a.set_xticks(range(len(names))); a.set_xticklabels([NAMES.get(n, n) for n in names], fontsize=7)
        a.set_title(lab)
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Test performance on held-out neighbourhoods (spatial blocks)")
    _save(fig, cfg, "03_model_comparison", demo)

    # 4 prediction maps + scatter
    obs, pred = maps["obs"], maps["pred"]
    lim = float(np.nanpercentile(np.abs(obs), 98))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    _img(ax[0], obs, "Observed LST anomaly", vmin=-lim, vmax=lim, label="deg C")
    _img(ax[1], pred, f"Predicted ({NAMES.get(best, best).replace(chr(10), ' ')})", vmin=-lim, vmax=lim, label="deg C")
    _img(ax[2], pred - obs, "Error (pred - obs)", cmap="PuOr_r", vmin=-3, vmax=3, label="deg C")
    for a in ax[:3]:
        a.contour(test_mask, levels=[0.5], colors="k", linewidths=0.6)
    m = test_mask & np.isfinite(obs)
    ax[3].hexbin(obs[m], pred[m], gridsize=50, cmap="viridis", mincnt=1, bins="log")
    lo, hi = np.percentile(obs[m], [0.5, 99.5])
    ax[3].plot([lo, hi], [lo, hi], "r--", lw=1); ax[3].set_xlim(lo, hi); ax[3].set_ylim(lo, hi)
    ax[3].set_xlabel("Observed (deg C)"); ax[3].set_ylabel("Predicted (deg C)"); ax[3].set_title("Test pixels (outlined blocks)")
    _save(fig, cfg, "04_prediction_maps", demo)

    # 5 permutation importance
    pi = summ["permutation_importance_rmse_increase"]
    order = sorted(pi, key=pi.get)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(order, [pi[k] for k in order], color="#1b9e77")
    ax.set_xlabel("Increase in test RMSE when shuffled (deg C)"); ax.set_title("Global feature importance (permutation)")
    _save(fig, cfg, "05_permutation_importance", demo)

    # 6 occlusion maps
    fig, ax = plt.subplots(1, len(GROUPS), figsize=(4 * len(GROUPS), 3.6))
    for i, g in enumerate(GROUPS):
        c = maps[f"contrib_{i}"]
        v = float(np.nanpercentile(np.abs(c), 98)) or 1
        _img(ax[i], c, g, cmap="RdBu_r", vmin=-v, vmax=v, label="deg C")
    fig.suptitle("Local explanations: how much each factor heats (+) or cools (-) each pixel")
    _save(fig, cfg, "06_occlusion_attribution", demo)

    # 7 cooling scenarios
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    v = float(np.nanpercentile(np.abs(np.concatenate([maps["cool_green"].ravel(), maps["cool_roof"].ravel()])), 99)) or 1
    _img(ax[0], maps["cool_green"], "Scenario: urban greening", cmap="Blues_r", vmin=-v, vmax=0, label="deg C change")
    _img(ax[1], maps["cool_roof"], "Scenario: cool roofs", cmap="Blues_r", vmin=-v, vmax=0, label="deg C change")
    fig.suptitle("Model-estimated surface cooling under what-if scenarios")
    _save(fig, cfg, "07_cooling_scenarios", demo)

    # 8 heat + trend
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    hv = float(np.nanpercentile(np.abs(maps["heat"]), 98))
    _img(ax[0], maps["heat"], f"Mean heat anomaly {summ['recent_years'][0]}-{summ['recent_years'][-1]}",
         vmin=-hv, vmax=hv, label="deg C")
    tv = float(np.nanpercentile(np.abs(maps["trend"][np.isfinite(maps["trend"])]), 98)) if np.isfinite(maps["trend"]).any() else 1
    _img(ax[1], maps["trend"], "Warming trend of anomaly", cmap="PuOr_r", vmin=-tv, vmax=tv, label="deg C / year")
    sy = [s for s in summ["suhi_by_year"] if s["suhi_c"] is not None]
    ax[2].plot([s["year"] for s in sy], [s["suhi_c"] for s in sy], "o-", color="#d95f02")
    ax[2].set_title("Surface urban heat island intensity\n(built-up minus vegetated)"); ax[2].set_ylabel("deg C")
    _save(fig, cfg, "08_heat_and_trend", demo)

    # forecasting figures
    fpath = cfg.out_dir / "metrics_forecast.json"
    if fpath.exists():
        fc = json.loads(fpath.read_text())
        fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
        for n, mm in fc["metrics"].items():
            ax[0].plot(range(1, len(mm["mae_by_lead"]) + 1), mm["mae_by_lead"], "o-", label=n.replace("_", " "))
        ax[0].set_xlabel("Lead time (days)"); ax[0].set_ylabel("MAE (deg C)"); ax[0].legend(fontsize=7)
        ax[0].set_title("Forecast error vs lead time (test years)")
        ex = fc["example"]
        x = np.arange(len(ex["dates"]))
        ax[1].plot(x, ex["observed_next_day"], "k-", lw=1.5, label="observed")
        ax[1].plot(x, ex["attention_lstm"], "-", color="#d95f02", label="attention LSTM")
        ax[1].plot(x, ex["persistence"], ":", color="grey", label="persistence")
        ax[1].axhline(fc["heat_stress_threshold_c"], color="red", lw=0.8, ls="--", label="heat-stress threshold")
        step = max(1, len(x) // 6)
        ax[1].set_xticks(x[::step]); ax[1].set_xticklabels([d[5:] for d in ex["dates"][::step]], fontsize=7)
        ax[1].set_title(f"Next-day max temperature, {ex['dates'][0][:4]} summer"); ax[1].legend(fontsize=7)
        att = fc["attention_by_lag"]
        ax[2].bar(range(-len(att) + 1, 1), att, color="#7570b3")
        ax[2].set_xlabel("Day relative to forecast date"); ax[2].set_title("Mean temporal attention weights")
        _save(fig, cfg, "09_forecasting", demo)
    print(f"Figures written to {cfg.fig_dir}")
