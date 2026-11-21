"""Differentiable greenhouse soil-water model on the probe's relative scale (0-100).

Dual-store formulation (v2, default) - hourly update for sector s:

  fast store (wetting pulse in the probe zone after irrigation, redistributes):
      f_{k+1} = (1 - r_s) f_k + alpha_s * I_k * S(theta_k)
  slow store (root-zone storage, FAO-56-style balance):
      s_{k+1} = s_k + beta_s r_s f_k - c_s ET0_k G(s_k) - d_s max(s_k - FC_s, 0)
  observation (capacitive probe with thermal sensitivity):
      theta_k = s_k + f_k + kappa_s (Tsoil_k - Tsoil_0)

  S(theta) = clip((SP - theta)/(SP - PWP), 0, 1)        saturating infiltration
  G(s)     = clip((s - PWP)/(MADP - PWP), 0, 1)          FAO-56 water-stress reduction (Ks)

PWP, MADP, FC, SP are FIXED from the agronomist (identifiability). Learned per sector:
alpha (pulse gain), r (redistribution rate), beta (fraction retained), c (lumped greenhouse ET
coefficient = cover transmissivity x Kc x points-per-mm), d (drainage), kappa (thermal sensitivity).
The fast store is initialised by running it over the previous NPAST hours of irrigation.
Rain is excluded (polytunnel). 'single' mode = classic one-store bucket (ablation).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config as C

NPAST = 12
BOUNDS = dict(alpha=(1e-3, 5.0), r=(0.01, 0.95), beta=(0.0, 1.0), c=(0.0, 30.0), d=(0.0, 0.5),
              kappa=(-3.0, 3.0))
INIT = dict(alpha=0.3, r=0.3, beta=0.3, c=2.0, d=0.05, kappa=0.0)


def _logit(v, lo, hi):
    x = (v - lo) / (hi - lo)
    return torch.logit(torch.tensor(float(x)).clamp(1e-4, 1 - 1e-4))


class Bucket(nn.Module):
    def __init__(self, z, init=None, mode="dual"):
        super().__init__()
        t = C.THRESH[z]
        self.PWP, self.MADP, self.FC, self.SP = t["PWP"], t["MADP"], t["FC"], t["SP"]
        self.mode = mode
        init = {**INIT, **(init or {})}
        self.raw = nn.ParameterDict({k: nn.Parameter(_logit(init[k], *BOUNDS[k])) for k in BOUNDS})

    def params(self):
        p = {k: BOUNDS[k][0] + (BOUNDS[k][1] - BOUNDS[k][0]) * torch.sigmoid(self.raw[k]) for k in BOUNDS}
        if self.mode == "single":
            p["r"] = torch.tensor(1.0); p["beta"] = torch.tensor(1.0); p["kappa"] = torch.tensor(0.0)
        return p

    def S(self, th):
        return ((self.SP - th) / (self.SP - self.PWP)).clamp(0, 1)

    def G(self, s):
        return ((s - self.PWP) / (self.MADP - self.PWP)).clamp(0, 1)

    def init_fast(self, th0, Ipast, p):
        f = torch.zeros_like(th0)
        for k in range(Ipast.shape[1]):
            f = (1 - p["r"]) * f + p["alpha"] * Ipast[:, k] * self.S(th0)
        return f

    def forward(self, th0, I, E, return_states=False):
        """th0 [B]; I [B, NPAST+H] liters (past then future); E [B, H, 2] = (ET0, dTsoil vs origin)."""
        p = self.params()
        Ip, If = I[:, :NPAST], I[:, NPAST:]
        f = self.init_fast(th0, Ip, p) if self.mode == "dual" else torch.zeros_like(th0)
        f = torch.minimum(f, th0.clamp(min=0))
        s = th0 - f
        out, fs = [], []
        for k in range(If.shape[1]):
            th = s + f
            nf = (1 - p["r"]) * f + p["alpha"] * If[:, k] * self.S(th)
            ns = s + p["beta"] * p["r"] * f - p["c"] * E[:, k, 0] * self.G(s) - p["d"] * (s - self.FC).clamp(min=0)
            if self.mode == "single":
                ns = s + p["alpha"] * If[:, k] * self.S(s) - p["c"] * E[:, k, 0] * self.G(s) - p["d"] * (s - self.FC).clamp(min=0)
                nf = torch.zeros_like(f)
            s, f = ns.clamp(0, 100), nf.clamp(0, 100)
            out.append((s + f + p["kappa"] * E[:, k, 1]).clamp(0, 100))
            fs.append(f)
        traj = torch.stack(out, 1)
        return (traj, torch.stack(fs, 1)) if return_states else traj


def load_et0():
    return pd.read_csv(C.PROCESSED / "era5_hourly_extra.csv", parse_dates=["ts"]).set_index("ts")


def arrays(h, ts, H=24):
    """Inputs for origins ts. h needs columns sm_f, sm, liters, et0, tsoil."""
    pos = {t: i for i, t in enumerate(h.index)}
    sm, obs = h["sm_f"].values, h["sm"].values
    lit = h["liters"].fillna(0).values
    et0 = h["et0"].fillna(0).values
    ts_ = h["tsoil"].interpolate(limit_area="inside").ffill().bfill().values
    keep, th0, I, E, Y = [], [], [], [], []
    for t in ts:
        i = pos.get(t)
        if i is None or i + H >= len(h) or i - NPAST + 1 < 0:
            continue
        keep.append(t)
        th0.append(sm[i])
        I.append(lit[i - NPAST + 1:i + H + 1])
        E.append(np.stack([et0[i + 1:i + H + 1], ts_[i + 1:i + H + 1] - ts_[i]], 1))
        Y.append(obs[i + 1:i + H + 1])
    f = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    return keep, f(th0), f(I), f(E), f(Y)


def fit(z, th0, I, E, Y, iters=300, lr=0.05, mode="dual"):
    m = Bucket(z, mode=mode)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    mask = ~torch.isnan(Y)
    Yz = torch.nan_to_num(Y)
    for _ in range(iters):
        loss = ((m(th0, I, E) - Yz).abs() * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return m, loss.item()


def params_dict(m):
    return {k: float(v.detach()) for k, v in m.params().items()}
