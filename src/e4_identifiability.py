"""E4: practical identifiability of the dual-store parameters by profile likelihood.
For each sector (fold-3 training data) and each parameter, fix it on a grid across its bounds,
refit all other parameters, and record the training MAE. A flat profile (< 2 % MAE change across
the range) means the parameter is practically non-identifiable."""
import numpy as np
import pandas as pd
import torch
import config as C
import e2_panel as P
import physics as PH
from e3_physics import hourly_with_et0, train_origins

GRID = 7


def fit_fixed(z, th0, I, E, Y, fixed, iters=150):
    m = PH.Bucket(z)
    with torch.no_grad():
        for k, v in fixed.items():
            m.raw[k].copy_(PH._logit(v, *PH.BOUNDS[k]))
    for k in fixed:
        m.raw[k].requires_grad_(False)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.05)
    mask = ~torch.isnan(Y)
    Yz = torch.nan_to_num(Y)
    for _ in range(iters):
        loss = ((m(th0, I, E) - Yz).abs() * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def main():
    torch.set_num_threads(4)
    rows = []
    for z in C.PRIMARY_SECTORS:
        h = hourly_with_et0(z)
        f = P.folds(z, h)[3]
        _, th0, I, E, Y = PH.arrays(h, train_origins(h, f, every=4))
        base = fit_fixed(z, th0, I, E, Y, {}, iters=300)
        for k, (lo, hi) in PH.BOUNDS.items():
            for v in np.linspace(lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo), GRID):
                rows.append(dict(sector=z, param=k, value=v, MAE=fit_fixed(z, th0, I, E, Y, {k: v}), base_MAE=base))
            print(f"S{z} {k} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(C.RESULTS / "e3" / "e4_profiles.csv", index=False)
    s = r.groupby(["sector", "param"]).agg(MAE_min=("MAE", "min"), MAE_max=("MAE", "max"), base=("base_MAE", "first"))
    s["range_%"] = 100 * (s.MAE_max - s.MAE_min) / s.MAE_min
    s["identifiable"] = s["range_%"] > 2
    s.to_csv(C.RESULTS / "e3" / "e4_identifiability.csv")
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
