"""PhyChronos-Irr forecaster: fault-aware, physics-guided fusion of Chronos-2 (+covariates) and the
differentiable greenhouse bucket.

For origin t, horizon h, quantile level a_i:
    m_h   = w_h(x) * c_med_h + (1 - w_h(x)) * p_h            # gated blend of FM median and physics
    q_i,h = m_h + s_h(x) * (c_i,h - c_med_h) + b_i,h         # FM spread, rescaled; quantile offsets
w (in [0,1]) and s (>0) come from a small MLP on reliability/physics features x:
    horizon, distance of theta_0 to agronomic thresholds, planned water, |c_med - p|,
    observation fraction (last 6 h), hours since last real reading, flatline length,
    6-h physics-consistency residual (fault score).
Trained per fold on training-period origins with pinball loss.
Usage: python e3_hybrid.py [--no-physics] [--no-fault] [--name NAME]
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config as C
import e2_panel as P
import physics as PH
from e3_physics import hourly_with_et0

H = P.H
Q = torch.tensor(P.QUANTILES)
CACHE = C.RESULTS / "e3" / "chronos2cov_all_origins.csv.gz"


def season_origins(z, h):
    a, b = P.season_window(z)
    ts = pd.date_range(a + pd.Timedelta(hours=P.ORIGIN_EVERY_H), b, freq=f"{P.ORIGIN_EVERY_H}h")
    sm = h["sm"]
    ok = []
    for t in ts:
        if pd.isna(sm.get(t)):
            continue
        fut = sm.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]
        if len(fut) == H and fut.notna().mean() >= C.MIN_TARGET_COVERAGE:
            ok.append(t)
    return ok


def chronos_all():
    if CACHE.exists():
        a = pd.read_csv(CACHE, parse_dates=["origin"])
        b = pd.read_csv(P.PRED_DIR / "Chronos-2+cov.csv.gz", parse_dates=["origin"])
        return pd.concat([a, b]).drop_duplicates(["sector", "origin", "h"]).reset_index(drop=True)
    from e2_foundation import run_chronos2, CTX
    items = []
    for z in C.PRIMARY_SECTORS:
        h = P.hourly(z)
        h["sm_f"] = P.sm_filled(h)
        for c in P.COVS_PAST:
            h[c] = h[c].interpolate(limit_area="inside").ffill().bfill()
        for t in season_origins(z, h):
            items.append((z, -1, t, h.loc[t - pd.Timedelta(hours=CTX - 1): t],
                          h.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]))
    df = pd.concat(run_chronos2(items, True), ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def fault_features(h, ts, bucket):
    """Reliability features at origins ts."""
    obs = h["sm_obs_frac"].fillna(0)
    real = h["sm_obs_frac"] > 0
    last_real = pd.Series(np.where(real, np.arange(len(h)), np.nan), index=h.index).ffill()
    since = pd.Series(np.arange(len(h)), index=h.index) - last_real
    sm = h["sm_f"]
    flat = sm.groupby((sm != sm.shift()).cumsum()).cumcount()
    # 6-h physics consistency residual
    keep6 = [t - pd.Timedelta(hours=6) for t in ts]
    k6, th0, I, E, _ = PH.arrays(h, keep6, H=6)
    with torch.no_grad():
        p6 = bucket(th0, I, E)[:, -1].numpy() if len(th0) else np.zeros(0)
    p6 = pd.Series(p6, index=[t + pd.Timedelta(hours=6) for t in k6]).reindex(ts).values
    resid = np.nan_to_num(np.abs(sm.reindex(ts).values - p6), nan=0.0)
    med3 = sm.shift(1).rolling(3, min_periods=1).median()
    spike = (sm - med3).abs().reindex(ts).fillna(0).values
    d24 = (sm - sm.shift(24)).reindex(ts).fillna(0).values
    sd6 = sm.rolling(6, min_periods=2).std().reindex(ts).fillna(0).values
    return np.stack([obs.rolling(6, min_periods=1).mean().reindex(ts).values,
                     np.minimum(since.reindex(ts).fillna(48).values, 48) / 48,
                     np.minimum(flat.reindex(ts).values, 48) / 48,
                     np.minimum(resid, 30) / 30,
                     np.minimum(spike, 30) / 30, np.clip(d24, -30, 30) / 30, np.minimum(sd6, 15) / 15], 1)


class Head(nn.Module):
    """Action-effect residual fusion.
    p[..., 0] = physics trajectory under the planned irrigation u, p[..., 1] = physics under u = 0.
        m = w * (c_med + g * (p_u - p_0)) + (1 - w) * p_u
    w in [0,1] (trust in the FM), g in [0, 1.5] (gain on the physics-predicted action effect)."""
    def __init__(self, nx, use_phys=True):
        super().__init__()
        self.use_phys = use_phys
        self.net = nn.Sequential(nn.Linear(nx + 1, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 3))
        self.b = nn.Parameter(torch.zeros(H, len(P.QUANTILES)))

    def forward(self, x, c, p):
        """x [B,nx], c [B,H,9] FM quantiles, p [B,H,2] physics (with / without planned water)."""
        B = x.shape[0]
        hh = torch.arange(1, H + 1, dtype=torch.float32).repeat(B, 1).unsqueeze(-1) / H
        xx = torch.cat([x.unsqueeze(1).expand(B, H, x.shape[1]), hh], -1)
        o = self.net(xx)
        cmed = c[..., 4]
        if self.use_phys:
            w = torch.sigmoid(o[..., 0] + 2.0)
            g = 1.5 * torch.sigmoid(o[..., 2])
            m = w * (cmed + g * (p[..., 0] - p[..., 1])) + (1 - w) * p[..., 0]
        else:
            w = torch.ones(B, H)
            m = cmed
        s = nn.functional.softplus(o[..., 1]) + 0.05
        q = m.unsqueeze(-1) + s.unsqueeze(-1) * (c - cmed.unsqueeze(-1)) + self.b
        return torch.sort(q, -1).values, w


def pinball(q, y):
    mask = ~torch.isnan(y)
    yz = torch.nan_to_num(y).unsqueeze(-1)
    e = yz - q
    l = torch.maximum(Q * e, (Q - 1) * e).mean(-1)
    return (l * mask).sum() / mask.sum()


ROBUST = True


def build_xy(z, h, ts, cdf, bucket, use_fault):
    keep, th0, I, E, Y = PH.arrays(h, ts)
    sub = cdf[cdf.sector == z].set_index(["origin", "h"])
    avail = set(sub.index.get_level_values(0))
    sel = [i for i, t in enumerate(keep) if t in avail]
    if not sel:
        return [], None, None, None, None
    keep = [keep[i] for i in sel]
    th0, I, E, Y = th0[sel], I[sel], E[sel], Y[sel]
    if ROBUST:   # spike-robust initial state for the physics branch: median of the last 3 hourly values
        med = h["sm_f"].rolling(3, min_periods=1).median()
        th0 = torch.tensor(med.reindex(keep).values, dtype=torch.float32)
    cq = np.stack([sub.loc[t][[f"q{a}" for a in P.QUANTILES]].values for t in keep]).astype(np.float32)
    I0 = I.clone()
    I0[:, PH.NPAST:] = 0
    with torch.no_grad():
        p = torch.stack([bucket(th0, I, E), bucket(th0, I0, E)], -1)
    t_ = C.THRESH[z]
    th = th0.numpy()
    x = [(th - t_["MADP"]) / 20, (th - t_["FC"]) / 20, (th - t_["PWP"]) / 20,
         I[:, PH.NPAST:].sum(1).numpy() / 50, np.abs(cq[:, :, 4] - p[..., 0].numpy()).mean(1) / 10,
         (p[..., 0] - p[..., 1]).mean(1).numpy() / 10]
    x = np.stack(x, 1)
    if use_fault:
        x = np.concatenate([x, fault_features(h, keep, bucket)], 1)
    return keep, torch.tensor(x, dtype=torch.float32), torch.tensor(cq), p, Y


def load_bucket(z, fold, mode="dual"):
    params = pd.read_csv(C.RESULTS / "e3" / f"params_{mode}.csv")
    pr = params[(params.sector == z) & (params.fold == fold)].iloc[0]
    return PH.Bucket(z, init={k: float(pr[k]) for k in PH.BOUNDS}, mode=mode)


def train_head(xtr, ctr, ptr, ytr, use_phys=True):
    torch.manual_seed(C.SEED)
    head = Head(xtr.shape[1], use_phys)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3, weight_decay=1e-4)
    n = len(xtr)
    cut = int(n * 0.85)
    best, state, bad = 1e9, None, 0
    for ep in range(400):
        head.train()
        q, _ = head(xtr[:cut], ctr[:cut], ptr[:cut])
        loss = pinball(q, ytr[:cut])
        opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vl = pinball(head(xtr[cut:], ctr[cut:], ptr[cut:])[0], ytr[cut:]).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad > 30:
                break
    head.load_state_dict(state)
    head.eval()
    return head, best


def main():
    use_phys = "--no-physics" not in sys.argv
    use_fault = "--no-fault" not in sys.argv
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "PhyChronos-Irr"
    torch.manual_seed(C.SEED)
    cdf = chronos_all()
    mode = "single" if "--single" in sys.argv else "dual"
    frames, gates = [], []
    for z, f, _, orig in P.all_tasks():
        if not orig:
            continue
        h = hourly_with_et0(z)
        bucket = load_bucket(z, f["fold"], mode)
        avail = set(cdf.loc[cdf.sector == z, "origin"])
        tr_ts = [t for t in season_origins(z, h) if t <= f["train_end"] - pd.Timedelta(hours=H) and t in avail]
        te_ts = [t for t in orig if t in avail]
        _, xtr, ctr, ptr, ytr = build_xy(z, h, tr_ts, cdf, bucket, use_fault)
        keep, xte, cte, pte, _ = build_xy(z, h, te_ts, cdf, bucket, use_fault)
        if not keep or xtr is None:
            continue
        n = len(xtr)
        head, best = train_head(xtr, ctr, ptr, ytr, use_phys)
        with torch.no_grad():
            q, w = head(xte, cte, pte)
        for t, qq, ww in zip(keep, q.numpy(), w.numpy()):
            frames.append(P.pred_frame(z, f["fold"], t, qq[:, 4], qq))
            gates.append(dict(sector=z, fold=f["fold"], origin=t, w_mean=float(ww.mean())))
        print(f"S{z} f{f['fold']}: n_train {n} val pinball {best:.3f} mean gate w(FM) {w.mean():.2f}", flush=True)
    P.save_preds(name, frames)
    pd.DataFrame(gates).to_csv(C.RESULTS / "e3" / f"gates_{name}.csv", index=False)


if __name__ == "__main__":
    main()
