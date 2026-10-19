"""Neural network architectures.

* UNet / AttentionUNet - fully-convolutional encoder-decoders that map a stack of
  Sentinel-2 bands, spectral indices and elevation to a per-pixel land-surface
  temperature anomaly. Skip connections keep fine spatial detail; the encoder gives
  the model neighbourhood context (a hot pixel is hotter when surrounded by concrete).
* AttnLSTMForecaster - LSTM encoder with additive temporal attention that forecasts
  daily maximum air temperature 1..7 days ahead from the last 30 days of weather.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018): the decoder's coarse signal decides
    which encoder features are passed through the skip connection."""

    def __init__(self, c_skip, c_gate, c_mid):
        super().__init__()
        self.ws = nn.Conv2d(c_skip, c_mid, 1)
        self.wg = nn.Conv2d(c_gate, c_mid, 1)
        self.psi = nn.Conv2d(c_mid, 1, 1)

    def forward(self, skip, gate):
        a = torch.sigmoid(self.psi(F.relu(self.ws(skip) + self.wg(gate))))
        self.last_attention = a.detach()
        return skip * a


class UNet(nn.Module):
    def __init__(self, in_ch=10, base=32, depth=3, attention=False, dropout=0.1):
        super().__init__()
        self.attention = attention
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.enc.append(ConvBlock(c, ch))
            c = ch
        self.bottleneck = ConvBlock(chs[-2], chs[-1], dropout)
        self.up, self.dec, self.gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(depth, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.gates.append(AttentionGate(chs[i - 1], chs[i - 1], chs[i - 1] // 2) if attention else nn.Identity())
            self.dec.append(ConvBlock(chs[i - 1] * 2, chs[i - 1]))
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, gate, dec, skip in zip(self.up, self.gates, self.dec, reversed(skips)):
            x = up(x)
            if self.attention:
                skip = gate(skip, x)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x).squeeze(1)


class PixelMLP(nn.Module):
    """Per-pixel network with no spatial context (1x1 convolutions) - ablation baseline
    that isolates how much the U-Net gains from looking at the neighbourhood."""

    def __init__(self, in_ch=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, hidden, 1),
                                 nn.ReLU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_lst_model(name: str, in_ch: int, base: int) -> nn.Module:
    if name == "unet":
        return UNet(in_ch, base, attention=False)
    if name == "attention_unet":
        return UNet(in_ch, base, attention=True)
    if name == "pixel_mlp":
        return PixelMLP(in_ch)
    raise ValueError(name)


class AttnLSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, horizon: int = 7):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.att_w = nn.Linear(hidden, hidden)
        self.att_v = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon))

    def forward(self, x, return_attention=False):
        h, _ = self.lstm(x)  # (B, T, H)
        scores = self.att_v(torch.tanh(self.att_w(h))).squeeze(-1)  # (B, T)
        a = torch.softmax(scores, dim=1)
        context = (a.unsqueeze(-1) * h).sum(dim=1)
        out = self.head(torch.cat([context, h[:, -1]], dim=1))
        return (out, a) if return_attention else out
