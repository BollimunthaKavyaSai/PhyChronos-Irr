"""Evaluate all prediction files in results/e2/preds against the hourly truth."""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import config as C
import e2_panel as P

OUT = C.RESULTS / "e2"


def load_truth():
    rows = []
    for z in C.PRIMARY_SECTORS:
        hh = P.hourly(z)
        h = hh["sm"]
        fut = hh["liters"].fillna(0)[::-1].rolling(H := P.H, min_periods=1).sum()[::-1].shift(-1)
        rows.append(pd.DataFrame({"sector": z, "ts": h.index, "y": h.values, "fut_w": fut.values}))
    return pd.concat(rows)


IRRIG = None   # set of (sector, origin) with planned water > 0 in the next 24 h


def wql(df):
    qcols = [f"q{q}" for q in P.QUANTILES]
    if qcols[0] not in df or df[qcols[0]].isna().all():
        return np.nan, np.nan, np.nan
    y = df["y"].values[:, None]
    Q = df[qcols].values
    a = np.array(P.QUANTILES)[None, :]
    pin = np.maximum(a * (y - Q), (a - 1) * (y - Q))
    w = 2 * pin.sum() / len(P.QUANTILES) / np.abs(y).sum()
    crps = 2 * pin.mean()                                   # quantile approx. of CRPS
    cov80 = ((y[:, 0] >= df["q0.1"]) & (y[:, 0] <= df["q0.9"])).mean()
    return w, crps, cov80


def evaluate(names=None):
    truth = load_truth()
    files = sorted(P.PRED_DIR.glob("*.csv.gz"))
    preds = {f.name[:-7]: pd.read_csv(f, parse_dates=["origin"]) for f in files}
    if names:
        preds = {k: v for k, v in preds.items() if k in names}
    # common origin set (intersection) so every model is scored on identical tasks
    keys = None
    for df in preds.values():
        k = set(map(tuple, df[["sector", "origin"]].drop_duplicates().values))
        keys = k if keys is None else keys & k
    tw = truth.set_index(["sector", "ts"])["fut_w"]
    irrig = {k for k in keys if tw.get(k, 0) > 0}
    rows, err_store = [], {}
    for name, df in preds.items():
        df = df[[tuple(x) in keys for x in df[["sector", "origin"]].values]].copy()
        df["ts"] = df["origin"] + pd.to_timedelta(df["h"], unit="h")
        df = df.merge(truth, on=["sector", "ts"], how="left").dropna(subset=["y"])
        df["ae"] = (df["point"] - df["y"]).abs()
        w, crps, cov = wql(df)
        m24 = df.groupby(["sector", "origin"]).agg(p=("point", "mean"), y=("y", "mean"))
        r = dict(model=name, n_origins=df.groupby(["sector", "origin"]).ngroups,
                 MAE_1_24=df["ae"].mean(), RMSE_1_24=np.sqrt(((df["point"] - df["y"]) ** 2).mean()),
                 MAE_h1=df.loc[df.h == 1, "ae"].mean(), MAE_h6=df.loc[df.h == 6, "ae"].mean(),
                 MAE_h12=df.loc[df.h == 12, "ae"].mean(), MAE_h24=df.loc[df.h == 24, "ae"].mean(),
                 MAE_mean24=(m24.p - m24.y).abs().mean(), WQL=w, CRPS=crps, Cov80=cov)
        isirr = np.array([(s, o) in irrig for s, o in zip(df.sector, df.origin)])
        r["MAE_irrig"] = df.loc[isirr, "ae"].mean()
        r["MAE_noirrig"] = df.loc[~isirr, "ae"].mean()
        for z in C.PRIMARY_SECTORS:
            r[f"MAE_S{z}"] = df.loc[df.sector == z, "ae"].mean()
        rows.append(r)
        err_store[name] = df.groupby(["sector", "origin"])["ae"].mean()
    res = pd.DataFrame(rows).sort_values("MAE_1_24")
    pers = res.set_index("model").loc["Persistence", "MAE_1_24"] if "Persistence" in res.model.values else np.nan
    res.insert(2, "Skill_vs_Pers", 1 - res["MAE_1_24"] / pers)
    # Diebold-Mariano-style test vs best model on per-origin MAE (HAC via block means by day)
    best = res.iloc[0]["model"]
    pv = []
    for name in res.model:
        if name == best:
            pv.append(np.nan); continue
        d = (err_store[name] - err_store[best]).dropna()
        d = d.groupby([d.index.get_level_values(0), d.index.get_level_values(1).floor("D")]).mean()
        t, p = stats.ttest_1samp(d, 0.0)
        pv.append(p)
    res["p_vs_best(daily-block)"] = pv
    return res, err_store


if __name__ == "__main__":
    res, _ = evaluate(sys.argv[1:] or None)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "e2_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
