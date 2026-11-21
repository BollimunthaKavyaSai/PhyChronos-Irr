"""E10: data efficiency. Train with the most recent 10/25/50/75/100 % of each fold's training
origins: PhyChronos-Irr fusion head and XGBoost-delta (point). Chronos-2+cov zero-shot needs no
training data and is shown as a constant reference."""
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
import config as C
import e2_panel as P
import e3_hybrid as HY
from e3_physics import hourly_with_et0
from e2_baselines import train_rows, features

H = P.H
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]


def main():
    cdf = HY.chronos_all()
    rows = []
    for z, f, _, orig in P.all_tasks():
        if not orig:
            continue
        h = hourly_with_et0(z)
        bucket = HY.load_bucket(z, f["fold"])
        avail = set(cdf.loc[cdf.sector == z, "origin"])
        tr_all = [t for t in HY.season_origins(z, h) if t <= f["train_end"] - pd.Timedelta(hours=H) and t in avail]
        te = [t for t in orig if t in avail]
        keep, xte, cte, pte, yte = HY.build_xy(z, h, te, cdf, bucket, True)
        if not keep:
            continue
        yn = yte.numpy()
        rows.append(dict(sector=z, fold=f["fold"], frac=np.nan, model="Chronos-2+cov (zero-shot)",
                         MAE=float(np.nanmean(np.abs(cte[:, :, 4].numpy() - yn)))))
        Xx, yx = train_rows(h, f)
        Xt = pd.concat([features(h, pd.DatetimeIndex([t] * H), np.arange(1, H + 1)) for t in keep])
        cur = np.repeat([h["sm_f"].loc[t] for t in keep], H)
        for fr in FRACS:
            sub = tr_all[-max(20, int(len(tr_all) * fr)):]
            _, xtr, ctr, ptr, ytr = HY.build_xy(z, h, sub, cdf, bucket, True)
            head, _ = HY.train_head(xtr, ctr, ptr, ytr)
            with torch.no_grad():
                q, _ = head(xte, cte, pte)
            rows.append(dict(sector=z, fold=f["fold"], frac=fr, model="PhyChronos-Irr",
                             MAE=float(np.nanmean(np.abs(q[:, :, 4].numpy() - yn)))))
            m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, verbosity=0)
            mask = _recent_mask(h, f, fr)
            m.fit(Xx[mask], yx[mask])
            px = m.predict(Xt) + cur
            rows.append(dict(sector=z, fold=f["fold"], frac=fr, model="XGBoost-delta",
                             MAE=float(np.nanmean(np.abs(px.reshape(len(keep), H) - yn)))))
        print(f"S{z} f{f['fold']} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(C.RESULTS / "e10_data_efficiency.csv", index=False)
    print(r.groupby(["model", "frac"], dropna=False)["MAE"].mean().round(3).to_string())


def _recent_mask(h, f, fr):
    """Boolean mask over train_rows() output keeping origins in the most recent fraction fr."""
    sm = h["sm"]
    ts = sm.loc[f["train_start"]:f["train_end"] - pd.Timedelta(hours=H)].dropna().index
    cut = ts[-max(20, int(len(ts) * fr))]
    masks = []
    for hh in range(1, H + 1):
        tgt = sm.reindex(ts + pd.Timedelta(hours=hh)).values
        ok = ~np.isnan(tgt)
        masks.append(ts[ok] >= cut)
    return np.concatenate(masks)


if __name__ == "__main__":
    main()
