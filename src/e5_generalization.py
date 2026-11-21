"""E5: generalisation, reported at separate levels.
 (a) Unseen sector (leave-one-sector-out): physics dynamics = median of the other sectors' fitted
     parameters (agronomic thresholds of the target sector are known); fusion head trained on the
     other sectors' in-season origins; evaluated on the held-out sector's E2 test origins.
     Compared with Chronos-2+cov zero-shot (no training at all) and persistence.
 (b) Regime shift: post-season period (irrigation stopped). Heads/physics trained on the whole season;
     evaluated on post-season origins every 6 h. XGBoost-delta (in-season trained) included.
"""
import numpy as np
import pandas as pd
import torch
import config as C
import e2_panel as P
import physics as PH
import e3_hybrid as HY
from e3_physics import hourly_with_et0, train_origins
from e2_foundation import run_chronos2, CTX

H = P.H
OUT = C.RESULTS / "e5"


def mae(p, y):
    return float(np.nanmean(np.abs(p - y)))


def loso(cdf):
    params = pd.read_csv(C.RESULTS / "e3" / "params_dual.csv")
    rows = []
    hs = {z: hourly_with_et0(z) for z in C.PRIMARY_SECTORS}
    tasks = list(P.all_tasks())
    for z_out in C.PRIMARY_SECTORS:
        others = [z for z in C.PRIMARY_SECTORS if z != z_out]
        med = params[params.sector.isin(others)][list(PH.BOUNDS)].median().to_dict()
        X, Cq, Pp, Y = [], [], [], []
        for z in others:
            b = PH.Bucket(z, init=med)
            avail = set(cdf.loc[cdf.sector == z, "origin"])
            ts = [t for t in HY.season_origins(z, hs[z]) if t in avail]
            _, x, c, p, y = HY.build_xy(z, hs[z], ts, cdf, b, True)
            X.append(x); Cq.append(c); Pp.append(p); Y.append(y)
        perm = torch.randperm(sum(len(x) for x in X), generator=torch.Generator().manual_seed(C.SEED))
        cat = lambda L: torch.cat(L)[perm]
        head, _ = HY.train_head(cat(X), cat(Cq), cat(Pp), cat(Y))
        b_out = PH.Bucket(z_out, init=med)
        for z, f, _, orig in tasks:
            if z != z_out or not orig:
                continue
            avail = set(cdf.loc[cdf.sector == z, "origin"])
            ts = [t for t in orig if t in avail]
            keep, x, c, p, y = HY.build_xy(z, hs[z], ts, cdf, b_out, True)
            with torch.no_grad():
                q, _ = head(x, c, p)
            yn = y.numpy()
            rows.append(dict(sector=z, fold=f["fold"], n=len(keep),
                             Persistence=mae(np.repeat(x.new_tensor([hs[z]["sm_f"].loc[t] for t in keep]).numpy()[:, None], H, 1), yn),
                             **{"Chronos-2+cov (zero-shot)": mae(c[:, :, 4].numpy(), yn),
                                "GH-DualStore (transferred)": mae(p[..., 0].numpy(), yn),
                                "PhyChronos-Irr (LOSO)": mae(q[:, :, 4].numpy(), yn)}))
        print(f"LOSO sector {z_out} done", flush=True)
    r = pd.DataFrame(rows)
    return r


def post_season():
    rows = []
    for z in C.PRIMARY_SECTORS:
        h = hourly_with_et0(z)
        a, b = P.season_window(z)
        ts = [t for t in pd.date_range(b + pd.Timedelta(hours=6), h.index[-1] - pd.Timedelta(hours=H), freq="6h")
              if pd.notna(h["sm"].get(t)) and h["sm"].loc[t + pd.Timedelta(hours=1):t + pd.Timedelta(hours=H)].notna().mean() >= 0.8]
        if len(ts) < 5:
            continue
        hh = h.copy()
        for c in P.COVS_PAST:
            hh[c] = hh[c].interpolate(limit_area="inside").ffill().bfill()
        items = [(z, -1, t, hh.loc[t - pd.Timedelta(hours=CTX - 1): t], hh.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]) for t in ts]
        cdf_post = pd.concat(run_chronos2(items, True), ignore_index=True)
        cdf_post["sector"] = z
        # physics + head trained on the full season (fold-3 bucket, all in-season origins)
        bucket = HY.load_bucket(z, 3)
        cdf_all = HY.chronos_all()
        avail = set(cdf_all.loc[cdf_all.sector == z, "origin"])
        tr = [t for t in HY.season_origins(z, h) if t in avail]
        _, xtr, ctr, ptr, ytr = HY.build_xy(z, h, tr, cdf_all, bucket, True)
        head, _ = HY.train_head(xtr, ctr, ptr, ytr)
        keep, x, c, p, y = HY.build_xy(z, h, ts, cdf_post, bucket, True)
        with torch.no_grad():
            q, w = head(x, c, p)
        # XGBoost-delta trained in-season (E2 configuration)
        from e2_baselines import train_rows, features
        from xgboost import XGBRegressor
        Xs, ys = train_rows(h, dict(train_start=a, train_end=b))
        xg = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, verbosity=0).fit(Xs, ys)
        px = np.stack([xg.predict(features(h, pd.DatetimeIndex([t] * H), np.arange(1, H + 1))) + h["sm_f"].loc[t] for t in keep])
        yn = y.numpy()
        pers = np.repeat(np.array([h["sm_f"].loc[t] for t in keep])[:, None], H, 1)
        rows.append(dict(sector=z, n=len(keep), Persistence=mae(pers, yn), **{
            "XGBoost-delta (in-season)": mae(px, yn), "Chronos-2+cov": mae(c[:, :, 4].numpy(), yn),
            "GH-DualStore": mae(p[..., 0].numpy(), yn), "PhyChronos-Irr": mae(q[:, :, 4].numpy(), yn),
            "mean gate w(FM)": float(w.mean())}))
        print(f"post-season sector {z} done ({len(keep)} origins)", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cdf = HY.chronos_all()
    a = loso(cdf)
    a.to_csv(OUT / "e5_loso.csv", index=False)
    cols = [c for c in a.columns if c not in ("sector", "fold", "n")]
    print("LOSO (MAE, mean over folds):\n", a.groupby("sector")[cols].mean().round(3).to_string())
    print(a[cols].mean().round(3).to_string())
    b = post_season()
    b.to_csv(OUT / "e5_postseason.csv", index=False)
    print("Post-season shift:\n", b.round(3).to_string(index=False))
