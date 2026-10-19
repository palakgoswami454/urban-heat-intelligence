"""Train and evaluate land-surface-temperature models (baselines + deep learning)."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CHANNELS, Config
from .dataset import PatchDataset, prepare, save_json
from .models import build_lst_model

DL_MODELS = ["pixel_mlp", "unet", "attention_unet"]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_loss(pred, y, m):
    m = m.float()
    loss = F.smooth_l1_loss(pred, y, reduction="none", beta=1.0)
    return (loss * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def predict_scene(model, Xn: np.ndarray, tile=256, overlap=32) -> np.ndarray:
    """Predict a full scene with overlapping tiles (border pixels are averaged)."""
    model.eval()
    C, H, W = Xn.shape
    step = tile - overlap
    out = np.zeros((H, W), "float32")
    cnt = np.zeros((H, W), "float32")
    for r in range(0, max(H - overlap, 1), step):
        for c in range(0, max(W - overlap, 1), step):
            r1, c1 = min(r + tile, H), min(c + tile, W)
            x = Xn[:, r:r1, c:c1]
            h, w = x.shape[1:]
            ph, pw = (-h) % 8, (-w) % 8
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(h, w) > 8 else "edge")
            p = model(torch.from_numpy(x)[None]).numpy()[0, :h, :w]
            out[r:r1, c:c1] += p
            cnt[r:r1, c:c1] += 1
    return out / np.maximum(cnt, 1)


def regression_metrics(y, p) -> dict:
    err = p - y
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "bias": float(err.mean()),
            "n_pixels": int(len(y))}


def hotspot_f1(obs_full, pred_full, eval_mask, q=90) -> dict:
    """Hotspot = pixel in the hottest 10% of its scene. F1 of predicted vs observed."""
    valid = np.isfinite(obs_full)
    t_obs = np.nanpercentile(obs_full[valid], q)
    t_pred = np.percentile(pred_full[valid], q)
    m = eval_mask & valid
    o, p = obs_full[m] >= t_obs, pred_full[m] >= t_pred
    tp = int((o & p).sum()); fp = int((~o & p).sum()); fn = int((o & ~p).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9), "tp": tp, "fp": fp, "fn": fn}


def evaluate_full(pred_maps, Y, test_mask) -> dict:
    ys, ps, f1s = [], [], []
    agg = {"tp": 0, "fp": 0, "fn": 0}
    for y, p in zip(Y, pred_maps):
        m = test_mask & np.isfinite(y)
        ys.append(y[m]); ps.append(p[m])
        h = hotspot_f1(y, p, test_mask)
        for k in agg:
            agg[k] += h[k]
    res = regression_metrics(np.concatenate(ys), np.concatenate(ps))
    prec = agg["tp"] / max(agg["tp"] + agg["fp"], 1)
    rec = agg["tp"] / max(agg["tp"] + agg["fn"], 1)
    res.update({"hotspot_precision": prec, "hotspot_recall": rec,
                "hotspot_f1": 2 * prec * rec / max(prec + rec, 1e-9)})
    return res


def pixel_samples(Xn, Y, mask, n, rng):
    feats, targs = [], []
    per = max(1, n // len(Xn))
    for x, y in zip(Xn, Y):
        idx = np.flatnonzero(mask & np.isfinite(y))
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(per, len(idx)), replace=False)
        feats.append(x.reshape(x.shape[0], -1)[:, idx].T)
        targs.append(y.reshape(-1)[idx])
    return np.concatenate(feats), np.concatenate(targs)


def train_baselines(Xn, Y, train_mask, test_mask, cfg) -> tuple[dict, dict]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(cfg.seed)
    Xtr, ytr = pixel_samples(Xn, Y, train_mask, 300_000, rng)
    results, maps = {}, {}
    zero_maps = [np.zeros_like(y) for y in Y]
    results["city_median (no model)"] = evaluate_full(zero_maps, Y, test_mask)
    for name, model in (("linear_regression", LinearRegression()),
                        ("gradient_boosting", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                                             random_state=cfg.seed))):
        t0 = time.time()
        model.fit(Xtr, ytr)
        pm = [model.predict(x.reshape(x.shape[0], -1).T).reshape(y.shape).astype("float32") for x, y in zip(Xn, Y)]
        results[name] = evaluate_full(pm, Y, test_mask)
        results[name]["train_seconds"] = round(time.time() - t0, 1)
        maps[name] = pm
        print(f"  {name:22s} test RMSE {results[name]['rmse']:.3f}  R2 {results[name]['r2']:.3f}")
    return results, maps


def train_dl(name, Xn, Y, idx, cfg, device):
    set_seed(cfg.seed)
    model = build_lst_model(name, len(CHANNELS), cfg.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tr = DataLoader(PatchDataset(Xn, Y, idx["train"], cfg.patch, augment=True), batch_size=cfg.batch_size,
                    shuffle=True, num_workers=0, drop_last=True)
    va = DataLoader(PatchDataset(Xn, Y, idx["val"], cfg.patch), batch_size=cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, wait, hist = 1e9, None, 0, []
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tl, tn = 0.0, 0
        for x, y, m in tr:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            loss = masked_loss(model(x), y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(x); tn += len(x)
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for x, y, m in va:
                p = model(x.to(device)).cpu()
                se += float((((p - y) ** 2) * m).sum()); n += int(m.sum())
        vrmse = (se / max(n, 1)) ** 0.5
        sched.step(vrmse)
        hist.append({"epoch": ep, "train_loss": tl / max(tn, 1), "val_rmse": vrmse,
                     "lr": opt.param_groups[0]["lr"]})
        print(f"  [{name}] epoch {ep:2d} train_loss {tl / max(tn, 1):.4f}  val_RMSE {vrmse:.3f}")
        if vrmse < best - 1e-4:
            best, wait = vrmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"  [{name}] early stop")
                break
    model.load_state_dict(best_state)
    model.cpu()
    return model, hist, {"params": n_params, "best_val_rmse": best, "train_seconds": round(time.time() - t0, 1),
                         "epochs_run": len(hist)}


def run(cfg: Config, models=DL_MODELS):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, torch.get_num_threads()))
    scenes, Xn, Y, blocks, split, norm, idx = prepare(cfg)
    train_mask = np.isin(blocks, split["train"])
    test_mask = np.isin(blocks, split["test"])
    save_json(norm, cfg.model_dir / "norm.json")
    save_json({"split": split, "n_blocks": cfg.n_blocks}, cfg.model_dir / "split.json")

    print("Baselines ...")
    results, _ = train_baselines(Xn, Y, train_mask, test_mask, cfg)
    histories, info = {}, {}
    for name in models:
        print(f"Training {name} on {device} ...")
        model, hist, meta = train_dl(name, Xn, Y, idx, cfg, device)
        torch.save({"state_dict": model.state_dict(), "name": name, "base": cfg.base_channels,
                    "channels": CHANNELS}, cfg.model_dir / f"lst_{name}.pt")
        pm = [predict_scene(model, x) for x in Xn]
        results[name] = evaluate_full(pm, Y, test_mask)
        results[name].update(meta)
        histories[name] = hist
        info[name] = meta
        print(f"  {name:22s} test RMSE {results[name]['rmse']:.3f}  R2 {results[name]['r2']:.3f}  "
              f"hotspot F1 {results[name]['hotspot_f1']:.3f}")

    dl_done = [m for m in models if m in results]
    best = min(dl_done, key=lambda m: info[m]["best_val_rmse"])  # chosen on validation, not test
    out = {"results": results, "best_model": best, "selection": "lowest validation RMSE",
           "target": "LST anomaly (deg C) relative to city-wide scene median",
           "split": {k: len(v) for k, v in split.items()}, "n_blocks": cfg.n_blocks,
           "scenes": [s.date for s in scenes], "n_scenes": len(scenes),
           "patches": {k: len(v) for k, v in idx.items()}, "grid_shape": list(scenes[0].lst.shape),
           "demo": cfg.demo}
    save_json(out, cfg.out_dir / "metrics_lst.json")
    save_json(histories, cfg.out_dir / "history_lst.json")
    print(f"Best model (validation): {best}")
    return out
