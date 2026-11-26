"""E13: backbone-agnostic BPT - equal-weight ensemble of all five zero-shot TSFM forecasts (point and quantiles averaged).
At the logged action, BPT returns its backbone forecast (q(pi_b) = q_FM), so the forecast accuracy of BPT with this
backbone equals the accuracy of the ensemble. Writes results/e2/preds/TSFM-Ensemble.csv.gz and
results/e2/e13_ensemble_stats.csv (paired moving-block bootstrap, sector x day blocks, B = 2000)."""
import numpy as np
import pandas as pd
import config as C
import e2_panel as P
from stats_bootstrap import boot

MEMBERS = ["Chronos-2", "TiRex", "Chronos-2+cov", "Moirai-2.0", "TimesFM-2.5"]
QC = [f"q{q}" for q in P.QUANTILES]
KEY = ["sector", "fold", "origin", "h"]


def main():
    D = {n: pd.read_csv(P.PRED_DIR / f"{n}.csv.gz", parse_dates=["origin"]).set_index(KEY) for n in MEMBERS}
    idx = D[MEMBERS[0]].index
    for n in MEMBERS[1:]:
        idx = idx.intersection(D[n].index)
    ens = pd.DataFrame(index=idx)
    for c in ["point"] + QC:
        ens[c] = np.mean([D[n].loc[idx, c].values for n in MEMBERS], axis=0)
    ens = ens.reset_index()
    ens.to_csv(P.PRED_DIR / "TSFM-Ensemble.csv.gz", index=False)

    truth = {z: P.hourly(z)["sm"] for z in C.PRIMARY_SECTORS}
    per = {}
    for n, d in list(D.items()) + [("TSFM-Ensemble", ens.set_index(KEY))]:
        d = d.reset_index()
        d["y"] = [truth[z].get(o + pd.Timedelta(hours=int(h))) for z, o, h in zip(d.sector, d.origin, d.h)]
        d = d.dropna(subset=["y"])
        per[n] = d.assign(ae=(d.point - d.y).abs()).groupby(["sector", "origin"])["ae"].mean()
    common = per["TSFM-Ensemble"].index
    blocks = np.array([f"{z}_{o.date()}" for z, o in common])
    rows = []
    for ref in MEMBERS:
        d = per["TSFM-Ensemble"].reindex(common).values - per[ref].reindex(common).values
        mu, lo, hi, p = boot(d, blocks)
        rows.append(dict(reference=ref, MAE_ens=per["TSFM-Ensemble"].mean(), MAE_ref=per[ref].reindex(common).mean(),
                         dMAE=mu, ci_lo=lo, ci_hi=hi, p=p))
    r = pd.DataFrame(rows)
    r.to_csv(C.RESULTS / "e2" / "e13_ensemble_stats.csv", index=False)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
