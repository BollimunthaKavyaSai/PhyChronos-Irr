"""E3/E4: fit the differentiable greenhouse soil-water model per sector and fold; physics-only
forecasts; parameters stored for identifiability analysis and the hybrid.
Usage: python e3_physics.py [single|dual]"""
import sys
import numpy as np
import pandas as pd
import torch
import config as C
import e2_panel as P
import physics as PH

H = P.H


def hourly_with_et0(z):
    h = P.hourly(z)
    h["sm_f"] = P.sm_filled(h)
    e = PH.load_et0()
    h["et0"] = e["et0_fao_evapotranspiration"].reindex(h.index).fillna(0)
    h["sw"] = e["shortwave_radiation"].reindex(h.index)
    h["vpd"] = e["vapour_pressure_deficit"].reindex(h.index)
    h["tsoil"] = h["soil_temperature_0-7cm"]
    return h


def train_origins(h, f, every=1):
    ts = h.loc[f["train_start"]:f["train_end"] - pd.Timedelta(hours=H)].index[::every]
    return ts[h["sm"].reindex(ts).notna().values]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dual"
    torch.manual_seed(C.SEED)
    torch.set_num_threads(4)
    frames, params = [], []
    for z, f, _, orig in P.all_tasks():
        h = hourly_with_et0(z)
        _, th0, I, E, Y = PH.arrays(h, train_origins(h, f, every=2))
        m, loss = PH.fit(z, th0, I, E, Y, mode=mode)
        p = PH.params_dict(m)
        params.append(dict(sector=z, fold=f["fold"], train_MAE=loss, n=len(th0), **p))
        if orig:
            keep, th0t, It, Et, _ = PH.arrays(h, orig)
            with torch.no_grad():
                pr = m(th0t, It, Et).numpy()
            for t, pp in zip(keep, pr):
                frames.append(P.pred_frame(z, f["fold"], t, pp))
        print(f"S{z} f{f['fold']}: train MAE {loss:.2f} " + " ".join(f"{k}={v:.3f}" for k, v in p.items()), flush=True)
    name = "GH-DualStore(physics)" if mode == "dual" else "GH-Bucket(physics)"
    P.save_preds(name, frames)
    out = C.RESULTS / "e3"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(params).to_csv(out / f"params_{mode}.csv", index=False)


if __name__ == "__main__":
    main()
