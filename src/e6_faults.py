"""E6: robustness to sensor faults occurring at forecast time (injected into the context only;
the truth is untouched). Origins: test origins spaced >= 48 h apart so perturbations never overlap.

Scenarios (applied to the 24 h before and at the origin):
  clean | dropout (no readings for 12 h -> deadband carry-forward) | stuck (frozen at t-24 value)
  drift (linear ramp to +8 points) | spike (+25 points at the origin)
Models: Persistence, Chronos-2+cov (zero-shot), GH-DualStore physics, PhyChronos-Irr (fault-aware),
PhyChronos-Irr without fault features.
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
SCEN = ["clean", "dropout", "stuck", "drift", "spike"]
OUT = C.RESULTS / "e6"


def spaced(orig, gap_h=24):
    out, last = [], None
    for t in sorted(orig):
        if last is None or (t - last) >= pd.Timedelta(hours=gap_h):
            out.append(t); last = t
    return out


def perturb(h, ts, scen):
    h = h.copy()
    for t in ts:
        w = h.loc[t - pd.Timedelta(hours=23): t].index
        if scen == "dropout":
            w12 = h.loc[t - pd.Timedelta(hours=11): t].index
            last = h["sm"].loc[:t - pd.Timedelta(hours=12)].dropna()
            v = last.iloc[-1] if len(last) else np.nan
            h.loc[w12, "sm"] = v; h.loc[w12, "sm_obs_frac"] = 0
        elif scen == "stuck":
            v = h["sm"].get(t - pd.Timedelta(hours=24))
            h.loc[w, "sm"] = v
        elif scen == "drift":
            h.loc[w, "sm"] = h.loc[w, "sm"] + np.linspace(0, 8, len(w))
        elif scen == "spike":
            h.loc[t, "sm"] = min(100, h.loc[t, "sm"] + 25)
    h["sm_f"] = P.sm_filled(h)
    return h


def chronos_preds(z, h, ts):
    hh = h.copy()
    for c in P.COVS_PAST:
        hh[c] = hh[c].interpolate(limit_area="inside").ffill().bfill()
    items = [(z, -1, t, hh.loc[t - pd.Timedelta(hours=CTX - 1): t].assign(sm_f=hh["sm_f"]),
              hh.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]) for t in ts]
    return pd.concat(run_chronos2(items, True), ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cdf_all = HY.chronos_all()
    rows = []
    for z, f, _, orig in P.all_tasks():
        if not orig:
            continue
        h0 = hourly_with_et0(z)
        bucket = HY.load_bucket(z, f["fold"])
        avail = set(cdf_all.loc[cdf_all.sector == z, "origin"])
        tr_ts = [t for t in HY.season_origins(z, h0) if t <= f["train_end"] - pd.Timedelta(hours=H) and t in avail]
        heads = {}
        for uf in (True, False):
            _, xtr, ctr, ptr, ytr = HY.build_xy(z, h0, tr_ts, cdf_all, bucket, uf)
            heads[uf], _ = HY.train_head(xtr, ctr, ptr, ytr)
        ts = spaced(orig)
        truth = {t: h0["sm"].loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)].values for t in ts}
        for scen in SCEN:
            h = perturb(h0, ts, scen)
            cdf = chronos_preds(z, h, ts)
            cdf["sector"] = z
            keep, th0, I, E, _ = PH.arrays(h, ts)
            with torch.no_grad():
                phys = bucket(th0, I, E).numpy()
            preds = {"Persistence": {t: np.full(H, h["sm_f"].loc[t]) for t in keep},
                     "Chronos-2+cov": {t: cdf[cdf.origin == t].sort_values("h")["point"].values for t in keep},
                     "GH-DualStore": dict(zip(keep, phys))}
            for uf, nm in ((True, "PhyChronos-Irr"), (False, "PhyChronos-Irr (no fault feat.)")):
                k2, x, c, p, _ = HY.build_xy(z, h, keep, cdf, bucket, uf)
                with torch.no_grad():
                    q, _ = heads[uf](x, c, p)
                preds[nm] = {t: qq[:, 4] for t, qq in zip(k2, q.numpy())}
            for m, d in preds.items():
                for t, p_ in d.items():
                    y = truth[t]
                    rows.append(dict(sector=z, fold=f["fold"], scenario=scen, model=m, origin=t,
                                     MAE=np.nanmean(np.abs(p_ - y))))
        print(f"S{z} f{f['fold']} done ({len(ts)} origins)", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "e6_raw.csv", index=False)
    tab = r.pivot_table(index="model", columns="scenario", values="MAE", aggfunc="mean")[SCEN]
    for s in SCEN[1:]:
        tab[f"deg_{s}_%"] = 100 * (tab[s] / tab["clean"] - 1)
    tab.to_csv(OUT / "e6_summary.csv")
    print(tab.round(2).to_string())


if __name__ == "__main__":
    main()
