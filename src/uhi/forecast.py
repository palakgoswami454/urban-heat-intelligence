"""Heat forecasting: Attention-LSTM predicting daily max temperature 1..7 days ahead.

Compared against persistence, climatology and ridge-regression baselines, and evaluated
on (a) temperature error per lead time and (b) detection of heat-stress days.
The trained weights are exported to JSON so the web dashboard can run the same model
live in the browser on the latest Open-Meteo data.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data_weather import DAILY_VARS
from .dataset import save_json
from .models import AttnLSTMForecaster

TARGET = "temperature_2m_max"


def make_features(df: pd.DataFrame) -> np.ndarray:
    doy = df.index.dayofyear.values
    return np.column_stack([df[v].values for v in DAILY_VARS] +
                           [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]).astype("float32")


def climatology(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    d = df[mask]
    clim = d.groupby(d.index.dayofyear)[TARGET].mean().reindex(range(1, 367)).interpolate().bfill().ffill().values
    ext = np.concatenate([clim[-15:], clim, clim[:15]])
    return np.convolve(ext, np.ones(15) / 15, mode="same")[15:-15].astype("float32")


def windows(F, tmax, T, Hh):
    idx = np.arange(T - 1, len(F) - Hh)
    X = np.stack([F[i - T + 1:i + 1] for i in idx])
    Y = np.stack([tmax[i + 1:i + 1 + Hh] for i in idx])
    return X, Y, idx


def f1_scores(y, p, thr):
    o, q = y >= thr, p >= thr
    tp, fp, fn = (o & q).sum(), (~o & q).sum(), (o & ~q).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"precision": float(prec), "recall": float(rec), "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
            "events": int(o.sum())}


def run(cfg: Config, df: pd.DataFrame):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    T, Hh = cfg.window, cfg.horizon
    Fr = make_features(df)
    tmax = df[TARGET].values.astype("float32")
    X, Y, idx = windows(Fr, tmax, T, Hh)
    dates = df.index[idx]
    tr = dates <= cfg.fc_train_end
    va = (dates > cfg.fc_train_end) & (dates <= cfg.fc_val_end)
    te = dates > cfg.fc_val_end

    f_mean = Fr[np.isin(np.arange(len(Fr)), idx[tr])].mean(0)
    f_std = Fr[np.isin(np.arange(len(Fr)), idx[tr])].std(0) + 1e-6
    Xn = (X - f_mean) / f_std
    last = tmax[idx][:, None]
    D = Y - last
    d_mean, d_std = D[tr].mean(0), D[tr].std(0) + 1e-6
    Dn = (D - d_mean) / d_std

    train_days = df.index <= cfg.fc_train_end
    clim = climatology(df, train_days)
    summer = df[train_days & df.index.month.isin([3, 4, 5])][TARGET]
    thr = float(np.percentile(summer, 90))

    # ---- baselines ----
    preds = {}
    preds["persistence"] = np.repeat(last, Hh, axis=1)
    doys = np.stack([df.index[idx + h].dayofyear.values for h in range(1, Hh + 1)], 1)
    preds["climatology"] = clim[doys - 1]
    from sklearn.linear_model import Ridge
    flat = Xn[:, -7:].reshape(len(Xn), -1)
    ridge = Ridge(alpha=1.0).fit(flat[tr], Dn[tr])
    preds["ridge_regression"] = last + ridge.predict(flat) * d_std + d_mean

    # ---- Attention-LSTM ----
    model = AttnLSTMForecaster(Xn.shape[2], cfg.lstm_hidden, Hh)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    tl = DataLoader(TensorDataset(torch.tensor(Xn[tr]), torch.tensor(Dn[tr])), batch_size=128, shuffle=True)
    Xv, Yv = torch.tensor(Xn[va]), Y[va]
    best, best_state, wait, hist = 1e9, None, 0, []
    for ep in range(1, cfg.fc_epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tl:
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            pv = model(Xv).numpy() * d_std + d_mean + last[va]
        vmae = float(np.abs(pv - Yv).mean())
        sched.step(vmae)
        hist.append({"epoch": ep, "train_loss": tot / tr.sum(), "val_mae": vmae})
        print(f"  [lstm] epoch {ep:2d} loss {tot / tr.sum():.4f} val MAE {vmae:.3f}")
        if vmae < best - 1e-4:
            best, wait = vmae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 6:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out, att = model(torch.tensor(Xn), return_attention=True)
    preds["attention_lstm"] = out.numpy() * d_std + d_mean + last
    att_mean = att.numpy()[te].mean(0)

    # ---- evaluation on the held-out test years ----
    metrics = {}
    for name, p in preds.items():
        e = p[te] - Y[te]
        metrics[name] = {
            "mae_by_lead": np.abs(e).mean(0).round(3).tolist(),
            "rmse_by_lead": np.sqrt((e ** 2).mean(0)).round(3).tolist(),
            "mae": float(np.abs(e).mean()),
            "heat_day": {f"lead_{h}": f1_scores(Y[te][:, h - 1], p[te][:, h - 1], thr) for h in (1, 3, 7)},
        }
        print(f"  {name:18s} test MAE {metrics[name]['mae']:.3f}  lead1 {metrics[name]['mae_by_lead'][0]:.2f}  "
              f"lead7 {metrics[name]['mae_by_lead'][-1]:.2f}")

    test_dates = dates[te]
    sample = (test_dates >= f"{test_dates.min().year}-03-01") & (test_dates <= f"{test_dates.min().year}-06-15")
    example = {"dates": [d.strftime("%Y-%m-%d") for d in test_dates[sample]],
               "observed_next_day": Y[te][sample][:, 0].round(2).tolist(),
               **{n: preds[n][te][sample][:, 0].round(2).tolist() for n in preds}}
    out = {"metrics": metrics, "heat_stress_threshold_c": round(thr, 2),
           "threshold_definition": "90th percentile of March-May daily max temperature in training years",
           "train_period": f"{df.index.min().date()} .. {cfg.fc_train_end}",
           "val_period": f"{cfg.fc_train_end} .. {cfg.fc_val_end}",
           "test_period": f"{cfg.fc_val_end} .. {df.index.max().date()}",
           "n_windows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
           "attention_by_lag": att_mean.round(4).tolist(), "history": hist, "example": example,
           "demo": cfg.demo}
    save_json(out, cfg.out_dir / "metrics_forecast.json")
    torch.save(model.state_dict(), cfg.model_dir / "forecast_lstm.pt")

    sd = {k: v.numpy().round(6).tolist() for k, v in model.state_dict().items()}
    web = {"features": DAILY_VARS + ["doy_sin", "doy_cos"], "window": T, "horizon": Hh,
           "hidden": cfg.lstm_hidden, "f_mean": f_mean.tolist(), "f_std": f_std.tolist(),
           "d_mean": d_mean.tolist(), "d_std": d_std.tolist(), "weights": sd,
           "climatology": clim.round(2).tolist(), "heat_threshold": round(thr, 2),
           "point": list(cfg.point), "test_mae": metrics["attention_lstm"]["mae_by_lead"],
           "persistence_mae": metrics["persistence"]["mae_by_lead"], "demo": cfg.demo}
    (cfg.web_data_dir / "forecast_model.json").write_text(json.dumps(web))
    # reference output for testing the JavaScript implementation
    ref_x = df.iloc[-T:]
    with torch.no_grad():
        xr = torch.tensor((Fr[-T:] - f_mean) / f_std)[None]
        pr = model(xr).numpy()[0] * d_std + d_mean + tmax[-1]
    save_json({"dates": [d.strftime("%Y-%m-%d") for d in ref_x.index],
               "inputs": {v: ref_x[v].round(4).tolist() for v in DAILY_VARS},
               "expected": pr.round(4).tolist()}, cfg.model_dir / "forecast_js_testcase.json")
    return out
