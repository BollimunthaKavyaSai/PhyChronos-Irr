"""Paired block bootstrap CIs (resampling whole blocks; block = sector x calendar day, B = 2000) for MAE differences.
E2: every model vs Chronos-2 (and vs PhyChronos-Irr-FA).  E6: PhyChronos-Irr-FA vs each model per fault scenario.
Writes results/stats_e2_bootstrap.csv, results/stats_e6_bootstrap.csv."""
import numpy as np
import pandas as pd
import config as C
import e2_panel as P

B = 2000
rng = np.random.default_rng(C.SEED)


def boot(diff, blocks):
    """diff: per-row paired differences; blocks: block id per row. Returns mean, lo, hi, p(two-sided)."""
    g = pd.Series(diff).groupby(blocks).agg(["sum", "count"])
    s, n = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), (B, len(g)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    m = s.sum() / n.sum()
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return m, np.quantile(bs, 0.025), np.quantile(bs, 0.975), max(p, 1 / B)


def e2():
    truth = {z: P.hourly(z)["sm"] for z in C.PRIMARY_SECTORS}
    per = {}
    for f in sorted(P.PRED_DIR.glob("*.csv.gz")):
        d = pd.read_csv(f, parse_dates=["origin"])
        d["y"] = [truth[z].get(o + pd.Timedelta(hours=int(h))) for z, o, h in zip(d.sector, d.origin, d.h)]
        d = d.dropna(subset=["y"])
        per[f.name[:-7]] = d.assign(ae=(d.point - d.y).abs()).groupby(["sector", "origin"])["ae"].mean()
    common = None
    for s in per.values():
        common = s.index if common is None else common.intersection(s.index)
    blocks = np.array([f"{z}_{o.date()}" for z, o in common])
    rows = []
    for ref in ["Chronos-2", "PhyChronos-Irr-FA"]:
        for m, s in per.items():
            if m == ref:
                continue
            d = s.reindex(common).values - per[ref].reindex(common).values
            mu, lo, hi, p = boot(d, blocks)
            rows.append(dict(model=m, reference=ref, MAE=s.reindex(common).mean(), dMAE=mu, ci_lo=lo, ci_hi=hi, p=p))
    r = pd.DataFrame(rows).sort_values(["reference", "MAE"])
    r.to_csv(C.RESULTS / "stats_e2_bootstrap.csv", index=False)
    print(r.round(3).to_string(index=False))


def e6():
    r = pd.read_csv(C.RESULTS / "e6" / "e6_raw.csv", parse_dates=["origin"])
    rows = []
    for scen, g in r.groupby("scenario"):
        w = g.pivot_table(index=["sector", "origin"], columns="model", values="MAE").dropna()
        blocks = np.array([f"{z}_{o.isocalendar().week}" for z, o in w.index])   # weekly blocks (origins 24 h apart)
        for m in w.columns:
            if m == "PhyChronos-Irr-FA":
                continue
            mu, lo, hi, p = boot(w["PhyChronos-Irr-FA"].values - w[m].values, blocks)
            rows.append(dict(scenario=scen, vs=m, n=len(w), MAE_FA=w["PhyChronos-Irr-FA"].mean(), MAE_other=w[m].mean(),
                             dMAE=mu, ci_lo=lo, ci_hi=hi, p=p))
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "stats_e6_bootstrap.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    e2()
    e6()
