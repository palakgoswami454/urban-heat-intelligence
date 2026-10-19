"""Explainability, counterfactual cooling scenarios, hotspot grid and recommendations.

Explainability
  * Permutation importance - shuffle a feature group across the map and measure how
    much the model's test error grows (global importance).
  * Occlusion attribution - replace a feature group with its training-average value and
    measure how the predicted temperature changes (local, per pixel, in deg C).
Counterfactual "what-if" scenarios (model-based estimates, not engineering guarantees)
  * Urban greening - blend 30% of a vegetation spectrum into impervious/bare pixels.
  * Cool roofs - raise the reflectance (albedo) of built-up pixels.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .config import CHANNELS, GROUPS, Config
from .data_satellite import spectral_indices
from .dataset import block_ids, load_scenes, normalize, save_json
from .grid import load_grid, pixel_to_lonlat
from .models import build_lst_model
from .train_lst import predict_scene

GROUP_IDX = {g: [CHANNELS.index(c) for c in cs] for g, cs in GROUPS.items()}


def load_best(cfg: Config):
    metrics = json.loads((cfg.out_dir / "metrics_lst.json").read_text())
    name = metrics["best_model"]
    ck = torch.load(cfg.model_dir / f"lst_{name}.pt", map_location="cpu")
    model = build_lst_model(name, len(CHANNELS), ck["base"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    norm = json.loads((cfg.model_dir / "norm.json").read_text())
    return model, norm, metrics


def permutation_importance(model, Xns, Ys, test_mask, seed=0, max_scenes=4):
    rng = np.random.default_rng(seed)
    sel = np.linspace(0, len(Xns) - 1, min(max_scenes, len(Xns))).round().astype(int)
    base_err, perm_err = [], {g: [] for g in GROUP_IDX}
    for i in sorted(set(sel)):
        x, y = Xns[i], Ys[i]
        m = test_mask & np.isfinite(y)
        p = predict_scene(model, x)
        base_err.append(((p[m] - y[m]) ** 2).mean())
        C, H, W = x.shape
        for g, ch in GROUP_IDX.items():
            xp = x.copy().reshape(C, -1)
            perm = rng.permutation(H * W)
            xp[ch] = xp[ch][:, perm]
            pp = predict_scene(model, xp.reshape(C, H, W))
            perm_err[g].append(((pp[m] - y[m]) ** 2).mean())
    base = float(np.sqrt(np.mean(base_err)))
    return {g: float(np.sqrt(np.mean(v)) - base) for g, v in perm_err.items()}, base


def occlusion(model, xn, base_pred):
    contrib = {}
    for g, ch in GROUP_IDX.items():
        xo = xn.copy()
        xo[ch] = 0.0  # 0 = training mean after normalisation
        contrib[g] = (base_pred - predict_scene(model, xo)).astype("float32")
    return contrib


def rebuild(s2, dem):
    return np.concatenate([s2, spectral_indices(s2), dem[None]], axis=0)


def scenario_greening(X, frac=0.3):
    s2, dem = X[:6].copy(), X[9]
    ndvi, mndwi = X[6], X[8]
    veg_mask = ndvi > 0.6
    if veg_mask.sum() < 50:
        veg_mask = ndvi >= np.nanpercentile(ndvi, 95)
    veg_spec = np.nanmedian(s2[:, veg_mask], axis=1)[:, None, None]
    target = (ndvi < 0.4) & (mndwi < 0)
    s2n = np.where(target[None], (1 - frac) * s2 + frac * veg_spec, s2)
    return rebuild(s2n, dem), target


def scenario_cool_roofs(X, d_vis_nir=0.15, d_swir=0.10):
    s2, dem = X[:6].copy(), X[9]
    built = (X[7] > 0) & (X[6] < 0.3) & (X[8] < 0)
    add = np.array([d_vis_nir] * 4 + [d_swir] * 2, dtype="float32")[:, None, None]
    s2n = np.where(built[None], np.clip(s2 + add, 0, 1), s2)
    return rebuild(s2n, dem), built


def block_reduce(a, k, fn=np.nanmean):
    H, W = a.shape
    ny, nx = -(-H // k), -(-W // k)
    pad = np.full((ny * k, nx * k), np.nan, dtype="float32")
    pad[:H, :W] = a
    v = pad.reshape(ny, k, nx, k).transpose(0, 2, 1, 3).reshape(ny, nx, k * k)
    with np.errstate(all="ignore"):
        return fn(v, axis=2)


def pct_rank(v):
    out = np.zeros_like(v, dtype="float64")
    ok = np.isfinite(v)
    if ok.sum() > 1:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    return out


def slope_per_year(years, M):
    """Least-squares trend (deg C / year) per pixel or cell from yearly mean maps M[y]."""
    years = np.asarray(years, dtype="float64")
    S = np.stack(M)  # (Y, ...)
    ok = np.isfinite(S)
    n = ok.sum(0)
    yb = np.where(ok, years[:, None, None], 0).sum(0) / np.maximum(n, 1)
    dy = np.where(ok, years[:, None, None] - yb, 0)
    vb = np.nansum(np.where(ok, S, 0), 0) / np.maximum(n, 1)
    dv = np.where(ok, S - vb, 0)
    with np.errstate(all="ignore"):
        sl = (dy * dv).sum(0) / (dy ** 2).sum(0)
    sl[n < 3] = np.nan
    return sl


RECS = {
    "water": "Water body - natural cooling; protect",
    "protect_green": "Existing green cover - protect from development",
    "greening": "Urban greening (street trees, parks, green cover)",
    "cool_roofs": "Cool / reflective roofs and pavements",
    "monitor": "Monitor - no strong single intervention signal",
}


def run(cfg: Config):
    model, norm, metrics = load_best(cfg)
    scenes = load_scenes(cfg)
    grid = load_grid(cfg.grid_path)
    split = json.loads((cfg.model_dir / "split.json").read_text())["split"]
    H, W = scenes[0].lst.shape
    blocks = block_ids(H, W, cfg.n_blocks)
    test_mask = np.isin(blocks, split["test"])
    Xns = [normalize(s.X, norm) for s in scenes]
    Ys = [s.anom for s in scenes]

    print("Permutation importance ...")
    perm, base_rmse = permutation_importance(model, Xns, Ys, test_mask, cfg.seed)

    # reference scene = most complete scene from the latest year available
    last_year = max(s.year for s in scenes)
    cand = [i for i, s in enumerate(scenes) if s.year >= last_year - 1]
    ref = max(cand, key=lambda i: (np.isfinite(scenes[i].lst).mean(), scenes[i].date))
    S, xn = scenes[ref], Xns[ref]
    print(f"Reference scene {S.date}: occlusion + scenarios ...")
    pred = predict_scene(model, xn)
    contrib = occlusion(model, xn, pred)
    Xg, tg = scenario_greening(S.X)
    Xr, tr = scenario_cool_roofs(S.X)
    cool_g = predict_scene(model, normalize(Xg, norm)) - pred
    cool_r = predict_scene(model, normalize(Xr, norm)) - pred
    cool_g[~tg] = 0.0
    cool_r[~tr] = 0.0

    # gap-filled anomaly per scene: observed where clear, model prediction under clouds
    print("Gap-filling + yearly maps ...")
    years = sorted({s.year for s in scenes})
    yearly = {}
    for y in years:
        maps = []
        for i, s in enumerate(scenes):
            if s.year != y:
                continue
            obs = s.anom
            filled = np.where(np.isfinite(obs), obs, predict_scene(model, Xns[i]))
            maps.append(filled)
        yearly[y] = np.nanmean(np.stack(maps), axis=0)
    recent_years = years[-3:]
    heat = np.nanmean(np.stack([yearly[y] for y in recent_years]), axis=0)
    trend_px = slope_per_year(years, [yearly[y] for y in years])

    # land-cover proxies from the reference scene
    ndvi, ndbi, mndwi = S.X[6], S.X[7], S.X[8]
    water = (mndwi > 0.1).astype("float32")
    green = (ndvi > 0.45).astype("float32")
    built = ((ndbi > 0) & (ndvi < 0.25) & (mndwi < 0)).astype("float32")

    # surface urban heat island intensity per year: built-up minus vegetated
    suhi = []
    for y in years:
        b = yearly[y][built > 0]; g = yearly[y][green > 0]
        suhi.append({"year": y, "suhi_c": float(np.nanmean(b) - np.nanmean(g)) if len(b) and len(g) else None})

    k = cfg.cell_px
    R = lambda a: block_reduce(a.astype("float32"), k)
    cell = {
        "heat": R(heat), "trend": block_reduce(trend_px.astype("float32"), k), "lst_ref": R(S.lst),
        "pred_ref": R(pred), "obs_ref": R(S.anom), "ndvi": R(ndvi), "ndbi": R(ndbi),
        "water": R(water), "green": R(green), "built": R(built),
        "cool_green": R(cool_g), "cool_roof": R(cool_r),
    }
    cell_contrib = {g: R(v) for g, v in contrib.items()}
    cell_yearly = {y: R(yearly[y]) for y in years}
    ny, nx = cell["heat"].shape

    valid = np.isfinite(cell["heat"])
    heat_v = cell["heat"][valid]
    hot_thr = max(float(np.nanpercentile(heat_v, 90)), 1.0)
    trend_pos = np.nan_to_num(np.clip(cell["trend"], 0, None))
    priority = 100 * (0.55 * pct_rank(cell["heat"]) + 0.30 * pct_rank(cell["built"]) + 0.15 * pct_rank(trend_pos))
    priority[(cell["water"] > 0.5) | ~valid] = 0

    rr = np.minimum(np.arange(ny + 1) * k, H)
    cc = np.minimum(np.arange(nx + 1) * k, W)
    RR, CC = np.meshgrid(rr, cc, indexing="ij")
    LON, LAT = pixel_to_lonlat(grid, RR, CC)
    cells = []
    for i in range(ny):
        for j in range(nx):
            if not valid[i, j]:
                continue
            ci = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
            lon = [LON[a, b] for a, b in ci]
            lat = [LAT[a, b] for a, b in ci]
            w, g, b = cell["water"][i, j], cell["green"][i, j], cell["built"][i, j]
            cg, cr = float(cell["cool_green"][i, j]), float(cell["cool_roof"][i, j])
            if w > 0.5:
                rec = "water"
            elif g > 0.5:
                rec = "protect_green"
            else:
                options = {"greening": cg}
                if b > 0.2:
                    options["cool_roofs"] = cr
                rec = min(options, key=options.get)
                if options[rec] > -0.3:
                    rec = "monitor"
            drivers = {gname: round(float(cell_contrib[gname][i, j]), 2) for gname in GROUPS}
            cells.append({
                "id": f"{i}-{j}",
                "c": [[round(float(la), 5), round(float(lo), 5)] for la, lo in zip(lat, lon)],
                "heat": round(float(cell["heat"][i, j]), 2),
                "trend": None if not np.isfinite(cell["trend"][i, j]) else round(float(cell["trend"][i, j]), 3),
                "lst": None if not np.isfinite(cell["lst_ref"][i, j]) else round(float(cell["lst_ref"][i, j]), 1),
                "pred": round(float(cell["pred_ref"][i, j]), 2),
                "obs": None if not np.isfinite(cell["obs_ref"][i, j]) else round(float(cell["obs_ref"][i, j]), 2),
                "ndvi": round(float(cell["ndvi"][i, j]), 3), "built": round(float(b), 3),
                "green": round(float(g), 3), "water": round(float(w), 3),
                "cool_green": round(cg, 2), "cool_roof": round(cr, 2),
                "priority": round(float(priority[i, j]), 1),
                "hot": bool(cell["heat"][i, j] >= hot_thr),
                "rec": rec, "drivers": drivers,
                "years": [None if not np.isfinite(cell_yearly[y][i, j]) else round(float(cell_yearly[y][i, j]), 2)
                          for y in years],
            })

    summary = {
        "reference_scene": S.date, "reference_median_lst_c": round(S.median, 2),
        "years": years, "recent_years": recent_years, "hotspot_threshold_c": round(hot_thr, 2),
        "n_cells": len(cells), "n_hotspots": int(sum(c["hot"] for c in cells)),
        "cell_size_m": k * grid["resolution"],
        "permutation_importance_rmse_increase": perm, "test_rmse_base": base_rmse,
        "suhi_by_year": suhi,
        "mean_cooling_greening_hotspots_c": float(np.mean([c["cool_green"] for c in cells if c["hot"]] or [0])),
        "mean_cooling_roofs_hotspots_c": float(np.mean([c["cool_roof"] for c in cells if c["hot"]] or [0])),
        "recommendation_counts": {r: sum(c["rec"] == r for c in cells if c["hot"]) for r in RECS},
    }
    save_json(summary, cfg.out_dir / "analysis_summary.json")
    np.savez_compressed(cfg.out_dir / "analysis_maps.npz", heat=heat, trend=trend_px, pred=pred,
                        obs=S.anom, lst=S.lst, cool_green=cool_g, cool_roof=cool_r, ndvi=ndvi,
                        **{f"contrib_{i}": v for i, v in enumerate(contrib.values())})
    (cfg.out_dir / "cells.json").write_text(json.dumps(cells))
    print(f"{len(cells)} grid cells, {summary['n_hotspots']} hotspots (>= {hot_thr:.2f} C above city median)")
    return summary, cells
