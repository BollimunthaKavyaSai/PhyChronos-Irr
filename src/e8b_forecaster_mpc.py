"""E8b: forecaster-in-the-loop irrigation control.  Same optimiser, different forecasters.

Hypothesis: a forecaster whose prediction responds correctly to the planned irrigation (action-aware)
makes better decisions than a more accurate but action-insensitive one.

Controller (identical for every forecaster): each day at 06:00 choose a daily dose D from
{0,30,60,120,180,240,360,480,720} min (delivered from 06 h at <= 60 min/h, i.e. within 06-18 h).  For each candidate the forecaster
returns 24-h quantiles; the lower bound L_h is the 10 % quantile (P(theta >= L_h) ~ 0.9).
Chance-constrained rule: minimise violation V(D) = sum_h relu(MADP - L_h); among candidates with
V <= min V + 0.5 choose the least water.
Forecasters: Physics (dual-store fitted on training data; L = point - one-sided split-conformal margin
from fold-0 residuals) | Chronos-2+cov (dose enters as future 'liters' covariate) | PhyChronos-Irr
(fusion head trained on fold-1 training origins) | PhyChronos-Delta (hybrid quantiles at u = 0 plus
the physics action effect p_u - p_0: data-driven baseline + mechanistic intervention effect).  Reference: Hysteresis (farm policy).
Plants A (physics) and B (XGBoost transition model) from e8_control.  Evaluation window = folds 1-3.
Usage: python e8b_forecaster_mpc.py [--noise]
"""
import sys
import numpy as np
import pandas as pd
import torch
import config as C
import e2_panel as P
import physics as PH
import e3_hybrid as HY
import e8_control as E8
from e3_physics import hourly_with_et0
from e2_foundation import CTX

H = 24
DOSES = [0, 30, 60, 120, 180, 240, 360, 480, 720]
NOISE = "--noise" in sys.argv
CTRLS = (sys.argv[sys.argv.index("--ctrls") + 1].split(",") if "--ctrls" in sys.argv
         else ["Hysteresis", "Physics", "Chronos-2", "PhyChronos-Irr", "PhyChronos-Delta", "Chronos-BPT", "RACER-Irr"])
KAPPA = float(sys.argv[sys.argv.index("--kappa") + 1]) if "--kappa" in sys.argv else 0.0   # extra safety margin (points)
PLANTS = sys.argv[sys.argv.index("--plants") + 1].split(",") if "--plants" in sys.argv else ["A", "B"]
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
OUT = C.RESULTS / "e8"
rng = np.random.default_rng(C.SEED)
_pipe = None


def pipe():
    global _pipe
    if _pipe is None:
        from chronos import Chronos2Pipeline
        _pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    return _pipe


def chronos_batch(hs, t, cands):
    """Chronos-2+cov quantiles [K,H,9] for candidate future-liters vectors."""
    ctx = hs.loc[t - pd.Timedelta(hours=CTX - 1): t]
    fut = hs.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]
    cr, fr = [], []
    for k, lit in enumerate(cands):
        c = pd.DataFrame({"item_id": str(k), "timestamp": ctx.index, "target": ctx["sm_f"].values})
        for cv in P.COVS_PAST:
            c[cv] = ctx[cv].values
        f = pd.DataFrame({"item_id": str(k), "timestamp": fut.index})
        for cv in P.COVS_FUTURE:
            f[cv] = lit if cv == "liters" else fut[cv].values
        cr.append(c); fr.append(f)
    out = pipe().predict_df(pd.concat(cr), future_df=pd.concat(fr), prediction_length=H,
                            quantile_levels=P.QUANTILES, validate_inputs=False)
    return np.stack([np.sort(out[out.item_id == str(k)].sort_values("timestamp")[[str(x) for x in P.QUANTILES]].values, 1)
                     for k in range(len(cands))]).astype(np.float32)


def phys_margin(z):
    d = pd.read_csv(P.PRED_DIR / "GH-DualStore(physics).csv.gz", parse_dates=["origin"])
    d = d[(d.sector == z) & (d.fold == 0)].copy()
    y = P.hourly(z)["sm"]
    d["y"] = y.reindex(d["origin"] + pd.to_timedelta(d["h"], unit="h")).values
    d = d.dropna(subset=["y"])
    return d.assign(r=d.point - d.y).groupby("h")["r"].quantile(0.9).clip(lower=0).reindex(range(1, H + 1)).fillna(0).values


def train_fold1_head(z, h, cdf, bucket, tr_end):
    avail = set(cdf.loc[cdf.sector == z, "origin"])
    tr = [t for t in HY.season_origins(z, h) if t <= tr_end - pd.Timedelta(hours=H) and t in avail]
    _, x, c, p, y = HY.build_xy(z, h, tr, cdf, bucket, True)
    return HY.train_head(x, c, p, y)[0]


def lower_bounds(kind, z, hs, t, cands, ctx):
    """Return [K,H] lower bounds and [K,H] medians for each candidate."""
    if kind == "Physics":
        out = []
        for lit in cands:
            hh = hs.copy()
            hh.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H), "liters"] = lit
            _, th0, I, E, _ = PH.arrays(hh, [t])
            with torch.no_grad():
                out.append(ctx["bucket"](th0, I, E)[0].numpy())
        med = np.array(out)
        return med - ctx["margin"][None], med
    if kind == "RACER-Irr":
        # counterfactual-trained head: Chronos at the no-irrigation plan, physics/feature paths per candidate
        cq = chronos_batch(hs, t, cands[:1])[0]
        cdf = P.pred_frame(z, 0, t, cq[:, 4], cq).assign(sector=z)
        lo, med = [], []
        for lit in cands:
            hh = hs.copy()
            hh.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H), "liters"] = lit
            keep, x, c, p, _ = HY.build_xy(z, hh, [t], cdf, ctx["bucket"], True)
            with torch.no_grad():
                q, _ = ctx["racer"](x, c, p)
            lo.append(q[0, :, 0].numpy()); med.append(q[0, :, 4].numpy())
        return np.array(lo), np.array(med)
    if kind == "Chronos-BPT":
        # Behaviour-Policy Transport: the FM forecast is read as the outcome under the logged (behaviour)
        # policy; physics transports it to candidate action u:  q(u) = q_FM + p(u) - p(u_beh).
        cq = chronos_batch(hs, t, cands[:1])[0]
        u_beh = behaviour_policy(z, hs, t, ctx)
        _, med_p = lower_bounds("Physics", z, hs, t, cands + [u_beh], ctx)
        shift = med_p[:-1] - med_p[-1:]
        return cq[None, :, 0] + shift, cq[None, :, 4] + shift
    if kind == "PhyChronos-Delta":
        cq = chronos_batch(hs, t, cands[:1])
        cdf = P.pred_frame(z, 0, t, cq[0][:, 4], cq[0]).assign(sector=z)
        hh = hs.copy()
        hh.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H), "liters"] = 0.0
        keep, x, c, p, _ = HY.build_xy(z, hh, [t], cdf, ctx["bucket"], True)
        with torch.no_grad():
            q0 = ctx["head"](x, c, p)[0][0].numpy()
        _, med_p = lower_bounds("Physics", z, hs, t, cands, ctx)
        eff = med_p - med_p[:1]
        return q0[None, :, 0] + eff, q0[None, :, 4] + eff
    cq = chronos_batch(hs, t, cands)
    if kind == "Chronos-2":
        return cq[:, :, 0], cq[:, :, 4]
    lo, med = [], []
    for k, lit in enumerate(cands):
        hh = hs.copy()
        hh.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H), "liters"] = lit
        cdf = P.pred_frame(z, 0, t, cq[k][:, 4], cq[k]).assign(sector=z)
        keep, x, c, p, _ = HY.build_xy(z, hh, [t], cdf, ctx["bucket"], True)
        with torch.no_grad():
            q, _ = ctx["head"](x, c, p)
        lo.append(q[0, :, 0].numpy()); med.append(q[0, :, 4].numpy())
    return np.array(lo), np.array(med)


def behaviour_policy(z, hs, t, ctx):
    """Roll the farm's logged hysteresis rule (start/stop thresholds, 06-18 h, 60 min/h) forward on the
    physics model from the current state; returns the implied 24-h liters vector."""
    t_ = C.THRESH[z]
    lit = np.zeros(H)
    hh = hs.copy()
    fut = hh.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)].index
    _, th0, I, E, _ = PH.arrays(hh, [t])
    th = float(th0[0])
    valve = bool(hh["liters"].get(t, 0) > 0)
    for j in range(H):
        hour = (t + pd.Timedelta(hours=j)).hour
        if th < t_["start"]:
            valve = True
        if th >= t_["stop"]:
            valve = False
        lit[j] = 60 * t_["flow"] if (valve and 6 <= hour < 18) else 0.0
        I2 = I.clone()
        I2[0, PH.NPAST:] = torch.tensor(lit, dtype=torch.float32)
        with torch.no_grad():
            th = float(ctx["bucket"](th0, I2, E)[0, j])
    return lit


def run(z, plant_kind, ctrl, h, plant_model, ctx):
    a, b = P.season_window(z)
    fl = P.folds(z, h)
    ev_start, ev_end = fl[1]["test_start"], fl[3]["test_end"]
    plant = E8.Plant(plant_kind, plant_model, z, NOISE)
    t_ = C.THRESH[z]
    flow = t_["flow"]
    idx = h.loc[ev_start:ev_end].index
    et0 = h["et0"].reindex(idx).fillna(0).values
    hs = h.copy()
    th = float(h["sm_f"].loc[ev_start])
    valve, day_min, log, decisions = False, 0, [], []
    for k, ts in enumerate(idx):
        ok = 6 <= ts.hour < 18
        if ctrl == "Fixed-interval":
            u = float(min(60, ctx["fixed_min"])) if ts.hour == 8 else 0.0
        elif ctrl == "EDML":
            if ts.hour == 6:
                from e2_baselines import features
                x = features(hs, pd.DatetimeIndex([ts]), np.array([H])).iloc[0]
                x["sm"] = th
                day_min = E8.edml_choose(ctx["edml"], x, flow, z)
            u = float(min(60, day_min)) if ok else 0.0
            day_min = max(0, day_min - u)
        elif ctrl == "Hysteresis":
            if th < t_["start"]:
                valve = True
            if th >= t_["stop"]:
                valve = False
            u = 60.0 if (valve and ok) else 0.0
        else:
            if ts.hour == 6 and k > 0:
                cands = []
                for D in DOSES:
                    lit = np.zeros(H)
                    rem = D
                    for j in range(H):            # future index j <-> sim step ts + j
                        m = min(60, rem); lit[j] = m * flow; rem -= m
                    cands.append(lit)
                lo, med = lower_bounds(ctrl, z, hs, ts, cands, ctx)
                V = np.maximum(t_["MADP"] + KAPPA - lo, 0).sum(1)
                okc = np.where(V <= V.min() + 0.5)[0]
                day_min = DOSES[int(okc.min())]
                decisions.append(dict(ts=ts, dose=day_min, V=float(V[okc.min()])))
            u = float(min(60, day_min)) if ok else 0.0
            day_min = max(0, day_min - u)
        liters = u * flow
        th_next = plant.step(th, liters, et0[k], ts.hour)
        # write the simulated history into the controller's view (probe = plant state)
        nxt = ts + pd.Timedelta(hours=1)
        if nxt in hs.index:
            hs.loc[nxt, ["sm", "sm_f"]] = th_next
            hs.loc[nxt, "liters"] = liters
            if "sm_obs_frac" in hs:
                hs.loc[nxt, "sm_obs_frac"] = 1.0
        log.append(dict(theta=th, liters=liters))
        th = th_next
    L = pd.DataFrame(log)
    return dict(sector=z, plant=plant_kind, controller=ctrl, noise=NOISE, kappa=KAPPA, water_L=L.liters.sum(),
                stress_h=int((L.theta < t_["MADP"]).sum()), danger_h=int((L.theta < t_["PWP"]).sum()),
                excess_h=int((L.theta > t_["FC"]).sum()),
                optimal_frac=float(((L.theta >= t_["MADP"]) & (L.theta < t_["FC"])).mean()),
                drainage=plant.drain, hours=len(L), n_decisions=len(decisions),
                mean_dose_min=float(np.mean([d["dose"] for d in decisions])) if decisions else np.nan)


PLANT_LOSO = "--plant-loso" in sys.argv
LOSO_FITS = {}


def main():
    torch.manual_seed(C.SEED)
    if PLANT_LOSO:
        for o in C.PRIMARY_SECTORS:
            ho = hourly_with_et0(o)
            ao, bo = P.season_window(o)
            LOSO_FITS[o] = E8.fit_bucket(o, ho, ao, bo)
            print("plant fit", o, {k: round(v, 3) for k, v in PH.params_dict(LOSO_FITS[o]).items()}, flush=True)
    cdf = HY.chronos_all()
    rows = []
    for z in C.PRIMARY_SECTORS:
        h = hourly_with_et0(z)
        for c in P.COVS_PAST:
            h[c] = h[c].interpolate(limit_area="inside").ffill().bfill()
        a, b = P.season_window(z)
        fl = P.folds(z, h)
        tr_end = fl[1]["train_end"]
        bucket = HY.load_bucket(z, 1)
        ctx = dict(bucket=bucket, margin=phys_margin(z))
        if any(c in CTRLS for c in ("PhyChronos-Irr", "PhyChronos-Delta")):
            ctx["head"] = train_fold1_head(z, h, cdf, bucket, tr_end)
        if "RACER-Irr" in CTRLS:
            import e12_racer as RC
            from e6b_fault_aug import build_cache
            lim = tr_end - pd.Timedelta(hours=H)
            avail = set(cdf.loc[cdf.sector == z, "origin"])
            tr = [t for t in HY.season_origins(z, h) if t <= lim and t in avail]
            X, Cq, Pp, Y, CF = RC.assemble(z, h, tr, cdf, build_cache(), bucket, lim)
            ctx["racer"] = RC.train_racer(X, Cq, Pp, Y, CF, z, True)
        if "EDML" in CTRLS:
            ctx["edml"] = E8.fit_edml(h, a, tr_end)
            ctx["fixed_min"] = h.loc[a:tr_end, "irr_min"].sum() / max((tr_end - a).days, 1)
        if PLANT_LOSO:
            # independently calibrated plant: median full-season parameters of the OTHER sectors,
            # with this sector's agronomic thresholds; no data of the target sector are used
            others = [PH.params_dict(LOSO_FITS[o]) for o in C.PRIMARY_SECTORS if o != z]
            med = {k: float(np.median([d[k] for d in others])) for k in others[0]}
            plants = {"A": PH.Bucket(z, init=med)}
        else:
            plants = {"A": E8.fit_bucket(z, h, a, b), "B": E8.fit_sim_b(h)}
        for pk, pm in plants.items():
            if pk not in PLANTS:
                continue
            for ctrl in CTRLS:
                r = run(z, pk, ctrl, h, pm, ctx)
                rows.append(r)
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                pd.DataFrame(rows).to_csv(OUT / f"e8b_results{'_noise' if NOISE else ''}{TAG}.csv", index=False)
    R = pd.DataFrame(rows)
    print(R.groupby(["plant", "controller"])[["water_L", "stress_h", "danger_h", "optimal_frac"]]
          .agg({"water_L": "sum", "stress_h": "sum", "danger_h": "sum", "optimal_frac": "mean"}).round(2).to_string())


if __name__ == "__main__":
    main()
