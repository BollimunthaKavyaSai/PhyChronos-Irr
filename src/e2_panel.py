"""Hourly forecasting panel, forecast origins and evaluation for E2+.

Task: at origin t (hour), forecast hourly mean soil moisture for t+1 .. t+H (H = 24).
Protocol (frozen in E1): active irrigation season, rolling-origin, 4 folds over the last
40 % of each sector's season, 24 h embargo between train end and test start.
"""
import numpy as np
import pandas as pd
import config as C

H = 24
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ORIGIN_EVERY_H = 3
COVS_PAST = ["weather_temp", "weather_humidity", "weather_radiation", "weather_wind_speed",
             "weather_rain", "liters", "ec"]
COVS_FUTURE = ["weather_temp", "weather_humidity", "weather_radiation", "weather_wind_speed",
               "weather_rain", "liters"]      # ERA5 as oracle weather; liters = planned irrigation
PRED_DIR = C.RESULTS / "e2" / "preds"


def season_window(z, pad_days=2):
    ev = pd.read_csv(C.EVENTS, parse_dates=["start", "end"])
    e = ev[ev.sector == z]
    return e["start"].min().normalize(), e["end"].max().normalize() + pd.Timedelta(days=pad_days)


def hourly(z):
    d = pd.read_csv(C.PROCESSED / f"zone_{z}.csv", parse_dates=["ts"]).set_index("ts")
    agg = {c: "mean" for c in ["weather_temp", "weather_humidity", "weather_radiation",
                               "weather_wind_speed", "weather_pressure", "soil_temperature_0-7cm",
                               "soil_temperature_7-18cm", "ec", "ph", "sm", "inside_temperature",
                               "inside_humidity"]}
    agg.update({"weather_rain": "sum", "liters": "sum", "irr_min": "sum", "sm_observed": "sum",
                "sm_fault": "sum"})
    h = d.resample("h").agg(agg)
    h["sm_obs_frac"] = h["sm_observed"] / 6
    h["sector"] = z
    return h


def sm_filled(h):
    """Gap-free version of sm for models that cannot take NaN (linear interp, edges ffill/bfill)."""
    return h["sm"].interpolate(limit_area="inside").ffill().bfill()


def folds(z, h, n_folds=4, test_frac=0.4, embargo_h=24):
    a, b = season_window(z)
    edges = pd.date_range(a + (b - a) * (1 - test_frac), b, periods=n_folds + 1).floor("h")
    out = []
    for k, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        out.append(dict(fold=k, train_start=a, train_end=s - pd.Timedelta(hours=embargo_h),
                        test_start=s, test_end=e))
    return out


def origins(z, h, fold):
    ts = pd.date_range(fold["test_start"], fold["test_end"] - pd.Timedelta(hours=1), freq=f"{ORIGIN_EVERY_H}h")
    ok = []
    sm = h["sm"]
    for t in ts:
        if t not in sm.index or pd.isna(sm.get(t)):
            continue
        fut = sm.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]
        if len(fut) == H and fut.notna().mean() >= C.MIN_TARGET_COVERAGE:
            ok.append(t)
    return ok


def all_tasks():
    """Yield (sector, fold dict, hourly frame, list of origins)."""
    for z in C.PRIMARY_SECTORS:
        h = hourly(z)
        for f in folds(z, h):
            yield z, f, h, origins(z, h, f)


def truth(h, t):
    return h["sm"].loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)].values


def pred_frame(z, fold, t, point, q=None):
    df = pd.DataFrame({"sector": z, "fold": fold, "origin": t, "h": np.arange(1, H + 1), "point": point})
    if q is not None:
        for i, ql in enumerate(QUANTILES):
            df[f"q{ql}"] = q[:, i]
    return df


def save_preds(name, frames):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(PRED_DIR / f"{name}.csv.gz", index=False)
