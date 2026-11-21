"""E2 baselines: persistence, diurnal-naive and quantile XGBoost (direct, horizon-as-feature, delta target)."""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import config as C
import e2_panel as P

H = P.H


def features(h, ts, hor):
    """Feature rows for origins ts and horizons hor (arrays of equal length)."""
    sm = P.sm_filled(h)
    roll = lambda s, w: s.rolling(w, min_periods=1)
    base = pd.DataFrame({
        "sm": sm, "sm_l1": sm.shift(1), "sm_l3": sm.shift(3), "sm_l6": sm.shift(6), "sm_l12": sm.shift(12),
        "sm_l24": sm.shift(24), "sm_m24": roll(sm, 24).mean(), "sm_s24": roll(sm, 24).std(),
        "w_past4": roll(h["liters"], 4).sum(), "w_past24": roll(h["liters"], 24).sum(),
        "temp": h["weather_temp"], "rh": h["weather_humidity"], "rad": h["weather_radiation"],
        "wind": h["weather_wind_speed"], "ec": h["ec"].ffill(),
    })
    X = base.reindex(ts).reset_index(drop=True)
    X["h"] = hor
    # future (known) covariates aggregated over (t, t+h]
    cum = {c: h[c].fillna(0).cumsum() for c in ["liters", "weather_rain", "weather_radiation", "weather_temp"]}
    t0 = pd.DatetimeIndex(ts)
    t1 = t0 + pd.to_timedelta(hor, unit="h")
    for c, s in cum.items():
        a = s.reindex(t0).values
        b = s.reindex(t1).values
        X[f"fut_{c}"] = (b - a) / (1 if c in ["liters", "weather_rain"] else np.asarray(hor))
    return X


def train_rows(h, fold):
    sm = h["sm"]
    ts = sm.loc[fold["train_start"]:fold["train_end"] - pd.Timedelta(hours=H)].dropna().index
    rows_t, rows_h, y = [], [], []
    for hh in range(1, H + 1):
        tgt = sm.reindex(ts + pd.Timedelta(hours=hh)).values
        ok = ~np.isnan(tgt)
        rows_t.append(ts[ok]); rows_h.append(np.full(ok.sum(), hh)); y.append(tgt[ok])
    ts_all = pd.DatetimeIndex(np.concatenate([r.values for r in rows_t]))
    hor = np.concatenate(rows_h)
    X = features(h, ts_all, hor)
    return X, np.concatenate(y) - X["sm"].values


def main():
    pers, diur, xgbq = [], [], []
    for z, f, h, orig in P.all_tasks():
        if not orig:
            continue
        sm = P.sm_filled(h)
        X, y = train_rows(h, f)
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(P.QUANTILES),
                         n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=-1, random_state=C.SEED, verbosity=0)
        m.fit(X, y)
        for t in orig:
            cur = sm.loc[t]
            pers.append(P.pred_frame(z, f["fold"], t, np.full(H, cur)))
            prof = sm.loc[t - pd.Timedelta(hours=23): t].values      # last 24 h profile
            diur.append(P.pred_frame(z, f["fold"], t, prof - prof[-1] + cur))
            Xt = features(h, pd.DatetimeIndex([t] * H), np.arange(1, H + 1))
            q = np.sort(m.predict(Xt), axis=1) + cur
            xgbq.append(P.pred_frame(z, f["fold"], t, q[:, 4], q))
        print(f"sector {z} fold {f['fold']} done ({len(orig)} origins, {len(y)} train rows)", flush=True)
    P.save_preds("Persistence", pers)
    P.save_preds("DiurnalNaive", diur)
    P.save_preds("XGBoost-Q", xgbq)


if __name__ == "__main__":
    main()
