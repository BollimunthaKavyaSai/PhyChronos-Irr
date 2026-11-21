"""E11: external validation on an independent farm (Mendeley cjb4vy4mzj, Jul-Sep 2022):
soil moisture + valve relay + cumulative flow meter. No site coordinates -> no weather/ET0, so the
physics branch cannot be parameterised; the test is zero-shot transfer of foundation models vs
trained baselines under the same rolling-origin protocol (last 40 %, 4 folds, 24 h embargo)."""
import glob
import numpy as np
import pandas as pd
import config as C
import e2_panel as P
import e2_foundation as F
from xgboost import XGBRegressor

H = 24
OUT = C.RESULTS / "e11"


def load():
    import os
    d = glob.glob(str(C.ROOT / "data/raw/secondary/cjb4vy4mzj/extracted/*"))[0] + os.sep
    fmt = "%d-%b-%Y %H:%M:%S"
    m = pd.read_csv(d + "D_moisture.txt", parse_dates=["time_stamp"], date_format=fmt).set_index("time_stamp")
    f = pd.read_csv(d + "D_flowmeter.txt", parse_dates=["time_stamp"], date_format=fmt).set_index("time_stamp")
    h = pd.DataFrame({"sm": m["moisture"].resample("h").mean()})
    lit = f["litres"].diff().clip(lower=0)
    lit[lit > 2000] = 0                                    # counter resets
    h["liters"] = lit.resample("h").sum().reindex(h.index).fillna(0)
    h["sm_f"] = h["sm"].interpolate(limit_area="inside").ffill().bfill()
    return h


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = load()
    t0, t1 = h.index[0], h.index[-1]
    edges = pd.date_range(t0 + (t1 - t0) * 0.6, t1 - pd.Timedelta(hours=H), periods=5).floor("h")
    P.COVS_PAST, P.COVS_FUTURE = ["liters"], ["liters"]
    items, folds = [], []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        for t in pd.date_range(a, b, freq="3h"):
            fut = h["sm"].loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]
            if pd.notna(h["sm"].get(t)) and len(fut) == H and fut.notna().mean() >= 0.8:
                items.append((0, k, t, h.loc[t - pd.Timedelta(hours=F.CTX - 1): t], h.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]))
        folds.append((k, a))
    preds = {"Chronos-2": F.run_chronos2(items, False), "Chronos-2+cov": F.run_chronos2(items, True),
             "TimesFM-2.5": F.run_timesfm(items), "Moirai-2.0": F.run_moirai2(items)}
    pers, diur, xg = [], [], []
    sm = h["sm_f"]
    for k, a in folds:
        tr = h.loc[:a - pd.Timedelta(hours=H)]
        X, Y = [], []
        for t in tr.index[48:-H]:
            if pd.isna(h["sm"].get(t)):
                continue
            for hh in (1, 3, 6, 12, 18, 24):
                y = h["sm"].get(t + pd.Timedelta(hours=hh))
                if pd.notna(y):
                    X.append(_feat(h, t, hh)); Y.append(y - sm[t])
        m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, verbosity=0).fit(np.array(X), np.array(Y))
        for it in [i for i in items if i[1] == k]:
            t = it[2]
            pers.append(P.pred_frame(0, k, t, np.full(H, sm[t])))
            prof = sm.loc[t - pd.Timedelta(hours=23): t].values
            diur.append(P.pred_frame(0, k, t, prof - prof[-1] + sm[t]))
            xg.append(P.pred_frame(0, k, t, m.predict(np.array([_feat(h, t, hh) for hh in range(1, H + 1)])) + sm[t]))
    preds.update({"Persistence": pers, "DiurnalNaive": diur, "XGBoost-delta": xg})
    rows = []
    for name, fr in preds.items():
        df = pd.concat(fr)
        df["y"] = h["sm"].reindex(df["origin"] + pd.to_timedelta(df["h"], unit="h")).values
        df = df.dropna(subset=["y"])
        ae = (df["point"] - df["y"]).abs()
        r = dict(model=name, n_origins=df.origin.nunique(), MAE_1_24=ae.mean(),
                 MAE_h1=ae[df.h == 1].mean(), MAE_h24=ae[df.h == 24].mean())
        if "q0.1" in df:
            r["Cov80"] = ((df.y >= df["q0.1"]) & (df.y <= df["q0.9"])).mean()
        rows.append(r)
    res = pd.DataFrame(rows).sort_values("MAE_1_24")
    res["Skill_vs_Pers"] = 1 - res.MAE_1_24 / res.set_index("model").loc["Persistence", "MAE_1_24"]
    res.to_csv(OUT / "e11_results.csv", index=False)
    print(res.round(3).to_string(index=False))


def _feat(h, t, hh):
    sm = h["sm_f"]
    g = lambda d: sm.get(t - pd.Timedelta(hours=d), sm[t])
    w = h["liters"]
    return [sm[t], g(1), g(3), g(6), g(12), g(24), hh, t.hour, w.loc[t - pd.Timedelta(hours=4):t].sum(),
            w.loc[t + pd.Timedelta(hours=1):t + pd.Timedelta(hours=hh)].sum()]


if __name__ == "__main__":
    main()
