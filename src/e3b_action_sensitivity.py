"""E3b: action sensitivity - does the forecast respond to a planned irrigation?

For test origins with no irrigation in the next 24 h, compare forecasts with a planned 30-min dose
starting at hour 2 versus no dose. Reported: mean peak response max_h(theta_dose - theta_nodose).
Empirical reference: observed rise after real events, measured as theta(start+2h) - theta(start)
minus the same difference in matched non-irrigated windows at the same hour (difference-in-differences).
"""
import numpy as np
import pandas as pd
import torch
import config as C
import e2_panel as P
import physics as PH
import e3_hybrid as HY
from e3_physics import hourly_with_et0
from e2_foundation import run_chronos2, CTX

H = P.H
DOSE_MIN, DOSE_H = 30, 2
OUT = C.RESULTS / "e3"


def empirical(z, h):
    ev = pd.read_csv(C.EVENTS, parse_dates=["start", "end"])
    e = ev[ev.sector == z]
    sm = h["sm"]
    rises = []
    for s in e["start"]:
        t0 = s.floor("h")
        a, b = sm.get(t0), sm.get(t0 + pd.Timedelta(hours=3))
        if pd.notna(a) and pd.notna(b):
            rises.append(b - a)
    irr = h["liters"].fillna(0).rolling(6, min_periods=1).sum()
    ctrl = []
    for t in sm.dropna().index:
        if 8 <= t.hour <= 12 and irr.get(t + pd.Timedelta(hours=3), 1) == 0 and irr.get(t, 1) == 0:
            b = sm.get(t + pd.Timedelta(hours=3))
            if pd.notna(b):
                ctrl.append(b - sm[t])
    return float(np.mean(rises) - np.mean(ctrl)), len(rises)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cdf_all = HY.chronos_all()
    rows = []
    for z, f, _, orig in P.all_tasks():
        if not orig or f["fold"] != 2:          # one representative fold per sector (keeps it fast)
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
        for dose in (0, DOSE_MIN):
            hd = h.copy()
            for t in ts:
                tt = t + pd.Timedelta(hours=DOSE_H)
                hd.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H), "liters"] = 0
                hd.loc[tt, "liters"] = dose * flow
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
        emp, n_ev = empirical(z, h)
        r = dict(sector=z, n_origins=len(ts), empirical_rise_3h=emp, n_events=n_ev)
        for m in ("chronos", "phys", "hybrid"):
            d = res[DOSE_MIN][m] - res[0][m]
            r[f"{m}_peak_response"] = float(d.max(1).mean())
            r[f"{m}_resp_3h"] = float(d[:, DOSE_H + 1].mean())
        rows.append(r)
        print(r, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e3b_action_sensitivity.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
