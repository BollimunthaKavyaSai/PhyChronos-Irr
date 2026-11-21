"""E1 - Reproduce the EDML/XGBoost soil-moisture regressor (Adamo et al. 2026) and
re-evaluate it under a corrected protocol.

Factorial design:  data {published, rebuilt}  x  protocol {SGKF-3day, rolling-origin+embargo}
plus persistence baseline and feature ablations.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
from xgboost import XGBRegressor
import config as C

OUT = C.RESULTS / "e1"
OUT.mkdir(parents=True, exist_ok=True)

# Feature set reported in the competitor's SHAP figure (15 features)
F15_PUB = ["ec", "soil_temperature_7-18cm", "soil_moisture", "water_vol_to_24h", "quarter", "weather_temp",
           "weather_pressure", "ph", "weather_wind_speed", "water_vol_past_4h", "weather_humidity",
           "soil_temperature_0-7cm", "weather_radiation", "weather_rain", "day_period"]
REN = {"soil_moisture": "sm", "water_vol_to_24h": "water_to_24h", "water_vol_past_4h": "water_past_4h"}
F15_REB = [REN.get(f, f) for f in F15_PUB]
CAL = ["quarter", "day_period"]
SENS = ["ec", "ph"]


def season_window(z, pad_days=2):
    ev = pd.read_csv(C.EVENTS, parse_dates=["start", "end"])
    e = ev[ev.sector == z]
    return e["start"].min().normalize(), e["end"].max().normalize() + pd.Timedelta(days=pad_days)


SEASON = False   # toggled in main()


def load(data, z):
    d = _load(data, z)
    if SEASON:
        a, b = season_window(z)
        d = d[(d.ts >= a) & (d.ts <= b)].reset_index(drop=True)
    return d


def _load(data, z):
    if data == "published":
        d = pd.read_csv(C.PUBLISHED / f"dataset_zone_{z}_preprocessed.csv", parse_dates=["ts"])
        d = d.rename(columns=REN).rename(columns={"target_mean_24h": "y"})
    else:
        d = pd.read_csv(C.PROCESSED / f"zone_{z}.csv", parse_dates=["ts"])
        d = d.rename(columns={"y_mean_24h": "y"})
        d = d[d["sm"].notna() & d["y"].notna()]
    return d.dropna(subset=["y", "sm"]).sort_values("ts").reset_index(drop=True)


def splits_sgkf(d, z):
    groups = ((d["ts"] - d["ts"].min()).dt.days // 3).values
    strata = C.agronomic_class(d["y"].values, z)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    return list(cv.split(d, strata, groups))


def splits_roll(d, n_folds=4, test_frac=0.4, embargo_h=24):
    t0, t1 = d["ts"].min(), d["ts"].max()
    start = t0 + (t1 - t0) * (1 - test_frac)
    edges = pd.date_range(start, t1, periods=n_folds + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.where(d["ts"] < a - pd.Timedelta(hours=embargo_h))[0]
        te = np.where((d["ts"] >= a) & (d["ts"] <= b))[0]
        if len(tr) > 200 and len(te) > 50:
            out.append((tr, te))
    return out


def fit_predict(d, feats, tr, te, delta=False):
    mono = tuple(1 if f == "water_to_24h" else 0 for f in feats)
    m = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, monotone_constraints=mono, n_jobs=-1,
                     random_state=C.SEED, verbosity=0)
    off_tr = d.loc[tr, "sm"].values if delta else 0
    off_te = d.loc[te, "sm"].values if delta else 0
    m.fit(d.loc[tr, feats], d.loc[tr, "y"] - off_tr)
    return m.predict(d.loc[te, feats]) + off_te


def metrics(y, p, z):
    cy, cp = C.agronomic_class(y, z), C.agronomic_class(p, z)
    return dict(MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5, R2=r2_score(y, p),
                F1_macro=f1_score(cy, cp, average="macro"), Acc=accuracy_score(cy, cp), n=len(y))


def run(data, protocol, model, feats, z):
    d = load(data, z)
    sp = splits_sgkf(d, z) if protocol == "SGKF-3day" else splits_roll(d)
    ys, ps, pers = [], [], []
    for tr, te in sp:
        ys.append(d.loc[te, "y"].values)
        pers.append(d.loc[te, "sm"].values)
        if model == "Persistence":
            ps.append(d.loc[te, "sm"].values)
        else:
            ps.append(fit_predict(d, feats, tr, te, delta=model.endswith("delta")))
    y, p, q = map(np.concatenate, (ys, ps, pers))
    r = metrics(y, p, z)
    r["MAE_skill_vs_persistence"] = 1 - r["MAE"] / mean_absolute_error(y, q)
    return dict(data=data, protocol=protocol, model=model, sector=z, **r)


def main():
    global SEASON
    import sys
    SEASON = "--season" in sys.argv
    tag = "season" if SEASON else "full"
    rows = []
    for z in C.SECTORS:
        for data in ["published", "rebuilt"]:
            F = F15_PUB if data == "published" else F15_REB
            F = [REN.get(f, f) for f in F]
            for protocol in ["SGKF-3day", "Rolling+embargo"]:
                rows.append(run(data, protocol, "Persistence", F, z))
                rows.append(run(data, protocol, "XGBoost", F, z))
                rows.append(run(data, protocol, "XGBoost-delta", F, z))
        # ablations on corrected setting
        base = F15_REB
        for name, feats in [("XGB -calendar", [f for f in base if f not in CAL]),
                            ("XGB -calendar -EC/pH", [f for f in base if f not in CAL + SENS]),
                            ("XGB -calendar -future water", [f for f in base if f not in CAL + ["water_to_24h"]])]:
            r = run("rebuilt", "Rolling+embargo", "XGBoost-delta", feats, z)
            r["model"] = name + " (delta)"
            rows.append(r)
        print(f"sector {z} done", flush=True)
    res = pd.DataFrame(rows)
    res["period"] = tag
    res.to_csv(OUT / f"e1_results_{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    show = res.copy()
    for c in ["MAE", "RMSE", "R2", "F1_macro", "Acc", "MAE_skill_vs_persistence"]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))
    prim = res[res.sector.isin(C.PRIMARY_SECTORS)].groupby(["data", "protocol", "model"])[
        ["MAE", "RMSE", "R2", "F1_macro", "MAE_skill_vs_persistence"]].mean().round(3)
    print("\nMean over primary sectors 1,2,4,5:\n", prim.to_string())
    prim.to_csv(OUT / f"e1_summary_primary_{tag}.csv")


if __name__ == "__main__":
    main()
