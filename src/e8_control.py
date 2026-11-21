"""E8: irrigation controllers evaluated by cross-model closed-loop simulation.

Plants (simulators, fitted on the full season, validated on held-out data first):
  SimA  physics: differentiable greenhouse bucket
  SimB  data-driven: XGBoost hourly transition model  d_theta = f(theta, water, ET0, hour, ...)
Controllers (hourly, irrigation allowed 06-18 h, per-sector emitter flow limit):
  Fixed-interval | Hysteresis (farm policy) | Predictive-threshold | EDML-style (XGB + exhaustive
  allocation, equivalent to the single-sector MILP on a 10-min grid) | D-MPC | CC-MPC (ours)
CC-MPC: receding horizon 24 h, replan every 3 h, model = bucket fitted on training data only,
chance constraint P(theta >= MADP) >= 0.9 enforced by one-sided split-conformal margins m_h computed
from PhyChronos-Irr calibration-fold residuals.
Scenario --noise: plant ET0 differs from the controller's ET0 forecast (+/-20 % daily multiplicative
error) and process noise sigma = 0.3 pt/h.
"""
import sys
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
import config as C
import e2_panel as P
import physics as PH
from e3_physics import hourly_with_et0

H = 24
OUT = C.RESULTS / "e8"
rng = np.random.default_rng(C.SEED)


# ------------------------------------------------------------------ simulators
def sim_b_features(th, liters, et0, hour, lit3):
    return np.column_stack([th, liters, et0, hour, lit3, th * et0, liters * (100 - th)])


def fit_sim_b(h, end=None):
    d = h if end is None else h.loc[:end]
    th = d["sm"].values
    dth = np.roll(th, -1) - th
    ok = ~np.isnan(th) & ~np.isnan(dth) & (d["sm_obs_frac"].values > 0)
    lit = d["liters"].fillna(0).values
    lit3 = pd.Series(lit).rolling(3, min_periods=1).sum().values
    X = sim_b_features(th, lit, d["et0"].values, d.index.hour.values, lit3)
    m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, verbosity=0,
                     random_state=C.SEED)
    m.fit(X[ok], dth[ok])
    return m


def fit_bucket(z, h, start, end):
    ts = h.loc[start:end - pd.Timedelta(hours=H)].index
    ts = ts[h["sm"].reindex(ts).notna().values]
    _, th0, I, E, Y = PH.arrays(h, ts)
    m, _ = PH.fit(z, th0, I, E, Y)
    return m


class Plant:
    def __init__(self, kind, model, z, noise):
        self.kind, self.m, self.z, self.noise = kind, model, z, noise
        self.lit_hist = [0.0, 0.0]
        self.drain = 0.0

    def step(self, th, liters, et0, hour):
        if self.kind == "A":
            # dual-store plant (true moisture, no probe thermal artefact)
            p = PH.params_dict(self.m)
            if not hasattr(self, "s"):
                self.s, self.f = th, 0.0
            S = np.clip((self.m.SP - th) / (self.m.SP - self.m.PWP), 0, 1)
            G = np.clip((self.s - self.m.PWP) / (self.m.MADP - self.m.PWP), 0, 1)
            dr = max(self.s - self.m.FC, 0) * p["d"]
            self.drain += dr + (1 - p["beta"]) * p["r"] * self.f
            nf = (1 - p["r"]) * self.f + p["alpha"] * liters * S
            ns = self.s + p["beta"] * p["r"] * self.f - p["c"] * et0 * G - dr
            self.s, self.f = float(np.clip(ns, 0, 100)), float(np.clip(nf, 0, 100))
            if self.noise:
                self.s += rng.normal(0, 0.3)
            return float(np.clip(self.s + self.f, 0, 100))
        else:
            lit3 = sum(self.lit_hist[-2:]) + liters
            nth = th + float(self.m.predict(sim_b_features(np.array([th]), np.array([liters]), np.array([et0]),
                                                           np.array([hour]), np.array([lit3])))[0])
            self.drain += max(nth - C.THRESH[self.z]["FC"], 0) * 0.05
        self.lit_hist.append(liters)
        if self.noise:
            nth += rng.normal(0, 0.3)
        return float(np.clip(nth, 0, 100))


# ------------------------------------------------------------------ controllers
def allowed(ts):
    return (ts.hour >= 6) & (ts.hour < 18)


class MPC:
    def __init__(self, z, bucket, margins, lam=50.0, mu=2.0):
        self.z, self.b, self.m, self.lam, self.mu = z, bucket, torch.tensor(margins, dtype=torch.float32), lam, mu
        self.t = C.THRESH[z]
        self.plan = None

    def solve(self, th, et0_fc, mask, flow, past_liters):
        u = torch.zeros(1, H, requires_grad=True)
        opt = torch.optim.Adam([u], lr=2.0)
        mk = torch.tensor(mask, dtype=torch.float32)[None]
        E = torch.zeros(1, H, 2)
        E[0, :, 0] = torch.tensor(et0_fc, dtype=torch.float32)
        past = torch.tensor(past_liters[-PH.NPAST:], dtype=torch.float32)[None]
        for _ in range(120):
            uu = u.clamp(0, 60) * mk
            traj = self.b(torch.tensor([th], dtype=torch.float32), torch.cat([past, uu * flow], 1), E)
            low = traj - self.m[None]
            loss = (uu.sum() * flow / 60 * 0.2 + self.lam * torch.relu(self.t["MADP"] - low).sum()
                    + self.mu * torch.relu(traj - self.t["FC"]).sum())
            opt.zero_grad(); loss.backward(); opt.step()
        return (u.detach().clamp(0, 60) * mk).numpy()[0]


def edml_choose(model, feats_row, flow, z):
    """Daily volume grid search: closest predicted 24-h mean to the Optimal-centre target, then least water."""
    t = C.THRESH[z]
    target = (t["MADP"] + t["FC"]) / 2
    best = None
    for minutes in range(0, 181, 10):
        x = feats_row.copy()
        x["fut_liters"] = minutes * flow
        y = float(model.predict(x[EDML_F].to_frame().T)[0])
        score = (abs(y - target), minutes)
        if best is None or score < best[0]:
            best = (score, minutes)
    return best[1]


EDML_F = ["sm", "sm_m24", "temp", "rh", "rad", "w_past4", "fut_liters", "fut_weather_radiation", "fut_weather_temp"]


def fit_edml(h, start, end):
    from e2_baselines import features
    sm = h["sm"]
    ts = sm.loc[start:end - pd.Timedelta(hours=H)].dropna().index
    X = features(h, ts, np.full(len(ts), H))
    y = sm.rolling(H).mean().shift(-H).reindex(ts).values
    ok = ~np.isnan(y)
    mono = tuple(1 if f == "fut_liters" else 0 for f in EDML_F)
    m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, monotone_constraints=mono, verbosity=0)
    m.fit(X.loc[ok, EDML_F], y[ok])
    return m


def conformal_margins(z):
    """One-sided split-conformal lower margins per horizon from PhyChronos-Irr fold-0 residuals."""
    f = P.PRED_DIR / "PhyChronos-Irr.csv.gz"
    if not f.exists():
        return np.zeros(H)
    d = pd.read_csv(f, parse_dates=["origin"])
    d = d[(d.sector == z) & (d.fold == 0)].copy()
    y = P.hourly(z)["sm"]
    d["y"] = y.reindex(d["origin"] + pd.to_timedelta(d["h"], unit="h")).values
    d = d.dropna(subset=["y"])
    return d.assign(r=(d["point"] - d["y"])).groupby("h")["r"].quantile(0.9).clip(lower=0).reindex(range(1, H + 1)).fillna(0).values


# ------------------------------------------------------------------ closed loop
def run(z, plant_kind, controller, noise):
    h = hourly_with_et0(z)
    a, b = P.season_window(z)
    fl = P.folds(z, h)
    ev_start, ev_end, tr_end = fl[1]["test_start"], fl[3]["test_end"], fl[1]["train_end"]
    plant_model = fit_bucket(z, h, a, b) if plant_kind == "A" else fit_sim_b(h)
    plant = Plant(plant_kind, plant_model, z, noise)
    t_ = C.THRESH[z]
    flow = t_["flow"]
    idx = h.loc[ev_start:ev_end].index
    et0_true = h["et0"].reindex(idx).fillna(0).values
    day_err = pd.Series(rng.normal(0, 0.2, len(idx) // 24 + 2)).repeat(24).values[:len(idx)] if noise else np.zeros(len(idx))
    et0_fc = h["et0"].fillna(0)
    th = float(h["sm_f"].loc[ev_start])
    ctrl_bucket = fit_bucket(z, h, a, tr_end) if controller in ("D-MPC", "CC-MPC") else None
    mpc = None
    if controller in ("D-MPC", "CC-MPC"):
        mpc = MPC(z, ctrl_bucket, conformal_margins(z) if controller == "CC-MPC" else np.zeros(H))
    edml = fit_edml(h, a, tr_end) if controller in ("EDML", "Predictive-threshold") else None
    fixed_min = h.loc[a:tr_end, "irr_min"].sum() / max((tr_end - a).days, 1)
    valve_on, plan, plan_t0, day_minutes = False, None, None, 0
    log = []
    for k, ts in enumerate(idx):
        ok = allowed(pd.DatetimeIndex([ts]))[0]
        u = 0.0
        if controller == "Fixed-interval":
            u = min(60, fixed_min) if ts.hour == 8 else 0
        elif controller == "Hysteresis":
            if th < t_["start"]:
                valve_on = True
            if th >= t_["stop"]:
                valve_on = False
            u = 60.0 if (valve_on and ok) else 0
        elif controller == "Predictive-threshold":
            if ok and ts.hour % 3 == 0:
                from e2_baselines import features
                x = features(h, pd.DatetimeIndex([ts]), np.array([H]))
                x["sm"] = th
                x["fut_liters"] = 0
                pred = float(edml.predict(x[EDML_F])[0])
                if th < t_["MADP"] or pred < t_["MADP"]:
                    day_minutes = 90
            u = min(60, day_minutes) if ok else 0
            day_minutes = max(0, day_minutes - u)
        elif controller == "EDML":
            if ts.hour == 6:
                from e2_baselines import features
                x = features(h, pd.DatetimeIndex([ts]), np.array([H])).iloc[0]
                x["sm"] = th
                day_minutes = edml_choose(edml, x, flow, z)
            u = min(60, day_minutes) if ok else 0
            day_minutes = max(0, day_minutes - u)
        else:  # MPC variants, replan every 3 h
            if plan is None or (k - plan_t0) >= 3:
                fut = pd.date_range(ts, periods=H, freq="h")
                e_fc = et0_fc.reindex(fut).fillna(0).values * (1 + (day_err[k] if noise else 0))
                past_l = [0.0] * PH.NPAST + [r["liters"] for r in log]
                plan, plan_t0 = mpc.solve(th, e_fc, allowed(fut).astype(float), flow, past_l), k
            u = float(plan[k - plan_t0]) if ok else 0.0
        liters = u * flow
        th_next = plant.step(th, liters, et0_true[k], ts.hour)
        log.append(dict(ts=ts, theta=th, minutes=u, liters=liters))
        th = th_next
    L = pd.DataFrame(log)
    return dict(sector=z, plant=plant_kind, controller=controller, noise=noise,
                water_L=L.liters.sum(), stress_h=int((L.theta < t_["MADP"]).sum()),
                danger_h=int((L.theta < t_["PWP"]).sum()), excess_h=int((L.theta > t_["FC"]).sum()),
                optimal_frac=float(((L.theta >= t_["MADP"]) & (L.theta < t_["FC"])).mean()),
                drainage=plant.drain, hours=len(L))


def validate_sims():
    rows = []
    for z in C.PRIMARY_SECTORS:
        h = hourly_with_et0(z)
        a, b = P.season_window(z)
        f3 = P.folds(z, h)[3]
        bA = fit_bucket(z, h, a, f3["train_end"])
        sB = fit_sim_b(h, f3["train_end"])
        ts = [t for t in P.origins(z, h, f3)]
        if not ts:
            continue
        keep, th0, I, E, Y = PH.arrays(h, ts)
        with torch.no_grad():
            pa = bA(th0, I, E).numpy()
        pb = []
        for i in range(len(keep)):
            th, lit_hist = float(th0[i]), [0.0, 0.0]
            tr = []
            for k in range(H):
                lit = float(I[i, PH.NPAST + k])
                X = sim_b_features(np.array([th]), np.array([lit]), np.array([float(E[i, k, 0])]),
                                   np.array([(keep[i] + pd.Timedelta(hours=k + 1)).hour]), np.array([sum(lit_hist[-2:]) + lit]))
                th = float(np.clip(th + sB.predict(X)[0], 0, 100)); lit_hist.append(lit); tr.append(th)
            pb.append(tr)
        Yn = Y.numpy()
        pers = np.repeat(th0.numpy()[:, None], H, 1)
        mae = lambda p: np.nanmean(np.abs(p - Yn))
        rows.append(dict(sector=z, n=len(keep), SimA_MAE=mae(pa), SimB_MAE=mae(np.array(pb)), Persistence_MAE=mae(pers)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--validate" in sys.argv:
        v = validate_sims()
        v.to_csv(OUT / "simulator_validation.csv", index=False)
        print(v.round(3).to_string(index=False))
        sys.exit()
    noise = "--noise" in sys.argv
    ctrls = ["Fixed-interval", "Hysteresis", "Predictive-threshold", "EDML", "D-MPC", "CC-MPC"]
    rows = []
    for z in C.PRIMARY_SECTORS:
        for plant in ["A", "B"]:
            for c in ctrls:
                r = run(z, plant, c, noise)
                rows.append(r)
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"e8_results{'_noise' if noise else ''}.csv", index=False)
    print(res.groupby(["plant", "controller"])[["water_L", "stress_h", "danger_h", "excess_h", "optimal_frac", "drainage"]].sum().round(2).to_string())
