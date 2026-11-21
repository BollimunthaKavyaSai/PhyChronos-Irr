"""E7: conformal calibration of 80 % intervals [q0.1, q0.9] for any probabilistic model.

Methods (per horizon h, per sector, processed in time order with DELAYED feedback: the score of
origin t at horizon h is only usable after t+h):
  * Uncalibrated
  * Split-CQR   : static offset from fold-0 scores (calibration fold), applied to folds 1-3
  * Rolling-CQR : offset = (1-alpha) empirical quantile of the last W revealed scores
  * ACI/QT      : online quantile tracking  delta <- delta + eta * (miss - alpha)   (Angelopoulos et al., 2023)
Evaluation on folds 1-3 only (fold 0 = calibration/burn-in for all methods).
"""
import sys
import numpy as np
import pandas as pd
import config as C
import e2_panel as P

ALPHA = 0.2
W = 120
OUT = C.RESULTS / "e7"


def load(name):
    df = pd.read_csv(P.PRED_DIR / f"{name}.csv.gz", parse_dates=["origin"])
    rows = []
    for z in C.PRIMARY_SECTORS:
        y = P.hourly(z)["sm"]
        d = df[df.sector == z].copy()
        d["ts"] = d["origin"] + pd.to_timedelta(d["h"], unit="h")
        d["y"] = y.reindex(d["ts"]).values
        rows.append(d)
    return pd.concat(rows).dropna(subset=["y"])


def cqr_score(d):
    return np.maximum(d["q0.1"] - d["y"], d["y"] - d["q0.9"])


def calibrate(df, method, eta=0.05):
    out = []
    for (z, hh), g in df.groupby(["sector", "h"]):
        g = g.sort_values("origin").copy()
        s = cqr_score(g).values
        reveal = (g["origin"] + pd.to_timedelta(hh, unit="h")).values
        org = g["origin"].values
        delta = np.zeros(len(g))
        if method == "Uncalibrated":
            pass
        elif method == "Split-CQR":
            cal = s[g["fold"].values == 0]
            delta[:] = np.quantile(cal, np.ceil((len(cal) + 1) * (1 - ALPHA)) / len(cal)) if len(cal) > 5 else 0
        else:
            d_t, j = 0.0, 0
            hist = []
            for i in range(len(g)):
                # absorb all scores revealed before this origin
                while j < len(g) and reveal[j] <= org[i]:
                    hist.append(s[j])
                    if method == "ACI-QT":
                        miss = float(s[j] > d_t)
                        d_t += eta * (miss - ALPHA) * (np.std(hist[-W:]) + 1e-3 if len(hist) > 5 else 1.0)
                    j += 1
                if method == "Rolling-CQR":
                    d_t = np.quantile(hist[-W:], 1 - ALPHA) if len(hist) >= 10 else 0.0
                delta[i] = d_t
        g["lo"] = g["q0.1"] - delta
        g["hi"] = g["q0.9"] + delta
        out.append(g)
    return pd.concat(out)


def summarize(name, methods=("Uncalibrated", "Split-CQR", "Rolling-CQR", "ACI-QT")):
    df = load(name)
    rows = []
    for m in methods:
        c = calibrate(df, m)
        c = c[c["fold"] > 0]
        cov = ((c.y >= c.lo) & (c.y <= c.hi))
        r = dict(model=name, method=m, coverage=cov.mean(), width=(c.hi - c.lo).mean(),
                 cov_h1_6=cov[c.h <= 6].mean(), cov_h19_24=cov[c.h >= 19].mean(),
                 worst_sector_cov=cov.groupby(c.sector).mean().min())
        # interval score (Gneiting & Raftery) for 80 %
        a = ALPHA
        r["interval_score"] = ((c.hi - c.lo) + 2 / a * (c.lo - c.y).clip(lower=0) + 2 / a * (c.y - c.hi).clip(lower=0)).mean()
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    names = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.concat([summarize(n) for n in names])
    res.to_csv(OUT / "e7_results.csv", index=False)
    print(res.round(3).to_string(index=False))
