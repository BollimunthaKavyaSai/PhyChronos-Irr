"""E6b: fault-augmented training of the PhyChronos-Irr fusion head (final model "PhyChronos-Irr-FA").

Augmentation: in-season origins are split into 8 phase groups (origins 24 h apart within a group,
so fault windows never overlap). Each group receives one synthetic fault type in its context
(dropout, stuck, drift, spike; two groups each). Chronos-2+cov is re-run on the corrupted contexts
(cached), physics/fault features are recomputed on the corrupted frame, labels stay CLEAN.
The head is trained on clean + augmented origins of each fold's training period and evaluated on
 (i) the clean E2 test origins (accuracy must not drop), (ii) the E6 fault scenarios.
"""
import numpy as np
import pandas as pd
import torch
import config as C
import e2_panel as P
import physics as PH
import e3_hybrid as HY
from e3_physics import hourly_with_et0
from e6_faults import perturb, spaced, chronos_preds, SCEN
from e2_foundation import run_chronos2, CTX

H = P.H
FAULTS = ["dropout", "stuck", "drift", "spike"]
CACHE = C.RESULTS / "e6" / "chronos_aug.csv.gz"


def phase_groups(z, h):
    S = HY.season_origins(z, h)
    t0 = S[0]
    groups = {p: [t for t in S if ((t - t0) // pd.Timedelta(hours=3)) % 8 == p] for p in range(8)}
    return groups


def build_cache():
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["origin"])
    frames = []
    for z in C.PRIMARY_SECTORS:
        h0 = hourly_with_et0(z)
        for p, ts in phase_groups(z, h0).items():
            hp = perturb(h0, ts, FAULTS[p % 4])
            c = chronos_preds(z, hp, ts)
            c["sector"], c["phase"] = z, p
            frames.append(c)
        print(f"aug cache sector {z} done", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(CACHE, index=False)
    return df


import sys
USE_PHYS = "--no-physics" not in sys.argv
NAME = "PhyChronos-Irr-FA" if USE_PHYS else "Abl-noPhysics-FA"
SCEN_CACHE = C.RESULTS / "e6" / "chronos_scen.csv.gz"


def main():
    torch.manual_seed(C.SEED)
    scache = pd.read_csv(SCEN_CACHE, parse_dates=["origin"]) if SCEN_CACHE.exists() else pd.DataFrame()
    new_cache, cov_rows = [], []
    cdf_clean = HY.chronos_all()
    aug = build_cache()
    e2_frames, e6_rows = [], []
    for z, f, _, orig in P.all_tasks():
        if not orig:
            continue
        h0 = hourly_with_et0(z)
        bucket = HY.load_bucket(z, f["fold"])
        lim = f["train_end"] - pd.Timedelta(hours=H)
        avail = set(cdf_clean.loc[cdf_clean.sector == z, "origin"])
        tr = [t for t in HY.season_origins(z, h0) if t <= lim and t in avail]
        parts = [HY.build_xy(z, h0, tr, cdf_clean, bucket, True)]
        for p, ts in phase_groups(z, h0).items():
            ts = [t for t in ts if t <= lim]
            if not ts:
                continue
            hp = perturb(h0, ts, FAULTS[p % 4])
            cp = aug[(aug.sector == z) & (aug.phase == p)]
            k, x, c, pp, _ = HY.build_xy(z, hp, ts, cp, bucket, True)
            if not k:
                continue
            Y = PH.arrays(h0, k)[4]                       # clean labels
            parts.append((k, x, c, pp, Y))
        X = torch.cat([q[1] for q in parts]); Cq = torch.cat([q[2] for q in parts])
        Pp = torch.cat([q[3] for q in parts]); Y = torch.cat([q[4] for q in parts])
        perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(C.SEED))
        head, _ = HY.train_head(X[perm], Cq[perm], Pp[perm], Y[perm], USE_PHYS)
        # (i) clean E2 test
        te = [t for t in orig if t in avail]
        k, x, c, pp, _ = HY.build_xy(z, h0, te, cdf_clean, bucket, True)
        if k:
            with torch.no_grad():
                q, _ = head(x, c, pp)
            for t, qq in zip(k, q.numpy()):
                e2_frames.append(P.pred_frame(z, f["fold"], t, qq[:, 4], qq))
        # (ii) E6 scenarios (same origins/protocol as e6_faults.py)
        ts6 = spaced(orig)
        truth = {t: h0["sm"].loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)].values for t in ts6}
        for scen in SCEN:
            hs = perturb(h0, ts6, scen)
            hit = scache[(scache.sector == z) & (scache.fold == f["fold"]) & (scache.scenario == scen)] if len(scache) else scache
            if len(hit):
                cs = hit.drop(columns=["scenario"]).copy()
            else:
                cs = chronos_preds(z, hs, ts6)
                cs["sector"], cs["fold"] = z, f["fold"]
                new_cache.append(cs.assign(scenario=scen))
            for t, g in cs.groupby("origin"):
                if t in truth:
                    y = truth[t]
                    cov_rows.append(dict(sector=z, fold=f["fold"], scenario=scen, model="Chronos-2+cov", origin=t,
                                         cov=float(np.nanmean((y >= g["q0.1"].values) & (y <= g["q0.9"].values))),
                                         width=float((g["q0.9"] - g["q0.1"]).mean())))
            k, x, c, pp, _ = HY.build_xy(z, hs, ts6, cs, bucket, True)
            with torch.no_grad():
                q, _ = head(x, c, pp)
            for t, qq in zip(k, q.numpy()):
                e6_rows.append(dict(sector=z, fold=f["fold"], scenario=scen, model=NAME, origin=t,
                                    MAE=float(np.nanmean(np.abs(qq[:, 4] - truth[t])))))
                cov_rows.append(dict(sector=z, fold=f["fold"], scenario=scen, model=NAME, origin=t,
                                     cov=float(np.nanmean((truth[t] >= qq[:, 0]) & (truth[t] <= qq[:, 8]))),
                                     width=float((qq[:, 8] - qq[:, 0]).mean())))
        print(f"S{z} f{f['fold']} done (train {len(X)})", flush=True)
    P.save_preds(NAME, e2_frames)
    if new_cache:
        pd.concat([scache] + new_cache).to_csv(SCEN_CACHE, index=False)
    cf = C.RESULTS / "e6" / "e6_coverage_raw.csv"
    cv = pd.DataFrame(cov_rows)
    if cf.exists():
        old_cv = pd.read_csv(cf, parse_dates=["origin"])
        cv = pd.concat([old_cv[~old_cv.model.isin(cv.model.unique())], cv])
    cv.to_csv(cf, index=False)
    print(cv.pivot_table(index="model", columns="scenario", values=["cov", "width"]).round(3).to_string())
    old = pd.read_csv(C.RESULTS / "e6" / "e6_raw.csv", parse_dates=["origin"])
    r = pd.concat([old[old.model != NAME], pd.DataFrame(e6_rows)])
    r.to_csv(C.RESULTS / "e6" / "e6_raw.csv", index=False)
    tab = r.pivot_table(index="model", columns="scenario", values="MAE", aggfunc="mean")[SCEN]
    for s in SCEN[1:]:
        tab[f"deg_{s}_%"] = 100 * (tab[s] / tab["clean"] - 1)
    tab.to_csv(C.RESULTS / "e6" / "e6_summary.csv")
    print(tab.round(2).to_string())


if __name__ == "__main__":
    main()
