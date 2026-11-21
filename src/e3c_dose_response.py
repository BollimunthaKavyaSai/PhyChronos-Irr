"""E3c: dose-response validation of action sensitivity.

Same origins/weather/initial state (fold-2 test origins without irrigation in the next 24 h); planned
dose D in {0,10,20,30,60,90} min at hour 2.  For each forecaster: mean response r(D) = theta_D - theta_0
at h = 3 (1 h after the dose) and the peak response; monotonicity = share of origins where the 3-h response is
non-decreasing in D.  Empirical reference: difference-in-differences 3-h rise after real events,
binned by event duration (same control windows as E3b).
"""
import numpy as np
import pandas as pd
import torch
import config as C
import e2_panel as P
import e3_hybrid as HY
from e3_physics import hourly_with_et0
from e2_foundation import run_chronos2, CTX
from e3b_action_sensitivity import empirical

H = P.H
DOSES = [0, 10, 20, 30, 60, 90]
DOSE_H = 2
OUT = C.RESULTS / "e3"


def empirical_by_duration(z, h):
    ev = pd.read_csv(C.EVENTS, parse_dates=["start", "end"])
    e = ev[ev.sector == z].copy()
    e["dur"] = (e["end"] - e["start"]).dt.total_seconds() / 60
    sm = h["sm"]
    irr = h["liters"].fillna(0).rolling(6, min_periods=1).sum()
    ctrl = [sm.get(t + pd.Timedelta(hours=3)) - sm[t] for t in sm.dropna().index
            if 8 <= t.hour <= 12 and irr.get(t + pd.Timedelta(hours=3), 1) == 0 and irr.get(t, 1) == 0]
    base = np.nanmean(ctrl)
    rows = []
    for lo, hi in [(0, 15), (15, 30), (30, 60), (60, 1e9)]:
        g = e[(e.dur >= lo) & (e.dur < hi)]
        r = [sm.get(s.floor("h") + pd.Timedelta(hours=3)) - sm.get(s.floor("h")) for s in g.start]
        r = [x for x in r if pd.notna(x)]
        rows.append(dict(sector=z, dur_bin=f"{lo:.0f}-{hi if hi < 1e8 else 'inf'}", n=len(r),
                         median_dur=float(g.dur.median()) if len(g) else np.nan,
                         did_rise=float(np.mean(r) - base) if r else np.nan))
    return rows


def main():
    cdf_all = HY.chronos_all()
    rows, emp_rows = [], []
    for z, f, _, orig in P.all_tasks():
        if not orig or f["fold"] != 2:
            continue
        h = hourly_with_et0(z)
        flow = C.THRESH[z]["flow"]
        fut_w = h["liters"].fillna(0)[::-1].rolling(H, min_periods=1).sum()[::-1].shift(-1)
        ts = [t for t in orig if fut_w.get(t, 1) == 0][:60]
        if len(ts) < 5:
            ts = orig[:60]
        bucket = HY.load_bucket(z, f["fold"])
        avail = set(cdf_all.loc[cdf_all.sector == z, "origin"])
        tr_ts = [t for t in HY.season_origins(z, h) if t <= f["train_end"] - pd.Timedelta(hours=H) and t in avail]
        _, xtr, ctr, ptr, ytr = HY.build_xy(z, h, tr_ts, cdf_all, bucket, True)
        head, _ = HY.train_head(xtr, ctr, ptr, ytr)
        res = {}
        for dose in DOSES:
            hd = h.copy()
            for t in ts:
                hd.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H), "liters"] = 0
                hd.loc[t + pd.Timedelta(hours=DOSE_H), "liters"] = dose * flow
            hh = hd.copy()
            for c in P.COVS_PAST:
                hh[c] = hh[c].interpolate(limit_area="inside").ffill().bfill()
            hh["sm_f"] = P.sm_filled(hh)
            items = [(z, -1, t, hh.loc[t - pd.Timedelta(hours=CTX - 1): t], hh.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]) for t in ts]
            cdf = pd.concat(run_chronos2(items, True), ignore_index=True)
            cdf["sector"] = z
            keep, x, c, p, _ = HY.build_xy(z, hd, ts, cdf, bucket, True)
            with torch.no_grad():
                q, _ = head(x, c, p)
            res[dose] = dict(chronos=c[:, :, 4].numpy(), phys=p[..., 0].numpy(), hybrid=q[:, :, 4].numpy())
        for m in ("chronos", "phys", "hybrid"):
            r3 = np.stack([res[d][m][:, DOSE_H] - res[0][m][:, DOSE_H] for d in DOSES], 1)   # [N, nD], h=3
            pk = np.stack([(res[d][m] - res[0][m]).max(1) for d in DOSES], 1)
            mono = float((np.diff(r3, axis=1) >= -0.05).all(1).mean())
            slope = float(np.polyfit(DOSES, r3.mean(0), 1)[0] * 30)       # points per 30 min
            for j, d in enumerate(DOSES):
                rows.append(dict(sector=z, model=m, dose=d, resp_3h=float(r3[:, j].mean()), peak=float(pk[:, j].mean()),
                                 monotone_frac=mono, slope_per30=slope))
        emp_rows += empirical_by_duration(z, h)
        print(f"sector {z} done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "e3c_dose_response.csv", index=False)
    E = pd.DataFrame(emp_rows)
    E.to_csv(OUT / "e3c_empirical_by_duration.csv", index=False)
    print(R.pivot_table(index=["sector", "model"], columns="dose", values="resp_3h").round(2).to_string())
    print(R.groupby(["sector", "model"])[["monotone_frac", "slope_per30"]].first().round(2).to_string())
    print(E.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
