"""Scene loading, spatial block cross-validation split, normalisation and patches."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CHANNELS, Config


@dataclass
class Scene:
    date: str
    X: np.ndarray  # (C, H, W) physical units, NaN where missing
    lst: np.ndarray  # (H, W) deg C, NaN where cloudy / missing
    median: float  # scene median LST (city-wide reference)
    source: str

    @property
    def anom(self) -> np.ndarray:
        """Surface urban heat anomaly: LST minus the scene's city-wide median."""
        return self.lst - self.median

    @property
    def year(self) -> int:
        return int(self.date[:4])


def load_scenes(cfg: Config) -> list[Scene]:
    scenes = []
    for p in sorted(cfg.scenes_dir.glob("scene_*.npz")):
        z = np.load(p, allow_pickle=False)
        X = z["X"].astype("float32")
        lst = z["lst"].astype("float32")
        valid = np.isfinite(lst) & np.isfinite(X).all(axis=0)
        lst[~valid] = np.nan
        scenes.append(Scene(str(z["date"]), X, lst, float(np.nanmedian(lst)), str(z["source"])))
    if not scenes:
        raise FileNotFoundError(f"No scenes in {cfg.scenes_dir}. Run the download stage first.")
    return scenes


def block_ids(H: int, W: int, n: int) -> np.ndarray:
    r = np.minimum((np.arange(H) * n) // H, n - 1)
    c = np.minimum((np.arange(W) * n) // W, n - 1)
    return r[:, None] * n + c[None, :]


def split_blocks(n: int, seed: int) -> dict:
    """Spatial hold-out: whole map blocks are reserved for validation and testing so the
    model is always evaluated on neighbourhoods it never saw during training."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n * n)
    n_test = max(1, round(0.25 * n * n))
    n_val = max(1, round(0.19 * n * n))
    return {"test": sorted(ids[:n_test].tolist()),
            "val": sorted(ids[n_test:n_test + n_val].tolist()),
            "train": sorted(ids[n_test + n_val:].tolist())}


def compute_norm(scenes: list[Scene], mask: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    samples = []
    for s in scenes:
        m = mask & np.isfinite(s.X).all(axis=0)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(len(idx), 50_000), replace=False)
        samples.append(s.X.reshape(len(CHANNELS), -1)[:, idx])
    S = np.concatenate(samples, axis=1)
    return {"mean": S.mean(axis=1).tolist(), "std": (S.std(axis=1) + 1e-6).tolist(), "channels": CHANNELS}


def normalize(X: np.ndarray, norm: dict) -> np.ndarray:
    m = np.asarray(norm["mean"], dtype="float32")[:, None, None]
    s = np.asarray(norm["std"], dtype="float32")[:, None, None]
    Z = (X - m) / s
    return np.nan_to_num(Z, nan=0.0).astype("float32")


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2))


def extract_patches(scene_idx: int, H: int, W: int, blocks: np.ndarray, keep_blocks: list,
                    target_valid: np.ndarray, patch: int, stride: int, min_valid=0.3):
    """Top-left corners of patches lying entirely inside one of `keep_blocks`."""
    out = []
    keep = set(keep_blocks)
    for r in range(0, H - patch + 1, stride):
        for c in range(0, W - patch + 1, stride):
            b = blocks[r:r + patch, c:c + patch]
            b0 = b[0, 0]
            if b0 not in keep or not (b == b0).all():
                continue
            if target_valid[r:r + patch, c:c + patch].mean() < min_valid:
                continue
            out.append((scene_idx, r, c))
    return out


class PatchDataset(Dataset):
    def __init__(self, Xn: list, Y: list, index: list, patch: int, augment: bool = False):
        self.Xn, self.Y, self.index, self.p, self.augment = Xn, Y, index, patch, augment

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, r, c = self.index[i]
        x = self.Xn[s][:, r:r + self.p, c:c + self.p]
        y = self.Y[s][r:r + self.p, c:c + self.p]
        m = np.isfinite(y)
        y = np.nan_to_num(y, nan=0.0)
        if self.augment:
            k = np.random.randint(4)
            x, y, m = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(m, k)
            if np.random.rand() < 0.5:
                x, y, m = x[:, :, ::-1], y[:, ::-1], m[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)).float(),
                torch.from_numpy(np.ascontiguousarray(m)))


def prepare(cfg: Config):
    """Load scenes and build normalised arrays, targets, block split and patch indices."""
    scenes = load_scenes(cfg)
    H, W = scenes[0].lst.shape
    blocks = block_ids(H, W, cfg.n_blocks)
    split = split_blocks(cfg.n_blocks, cfg.seed)
    train_mask = np.isin(blocks, split["train"])
    norm = compute_norm(scenes, train_mask, cfg.seed)
    Xn = [normalize(s.X, norm) for s in scenes]
    Y = [s.anom.astype("float32") for s in scenes]
    idx = {}
    for name, stride in (("train", cfg.train_stride), ("val", cfg.patch), ("test", cfg.patch)):
        idx[name] = []
        for i, s in enumerate(scenes):
            idx[name] += extract_patches(i, H, W, blocks, split[name], np.isfinite(Y[i]),
                                         cfg.patch, stride)
    print(f"Scenes: {len(scenes)} | grid {H}x{W} | patches train {len(idx['train'])} "
          f"val {len(idx['val'])} test {len(idx['test'])}")
    return scenes, Xn, Y, blocks, split, norm, idx
