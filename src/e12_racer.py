"""E12: RACER-Irr - Reliability-Aware Counterfactual Event-Response learning for irrigation forecasting.

Backbone: Chronos-2+cov (frozen, zero-shot) + differentiable dual-store physics (frozen, fitted per fold).
Learner: the reliability-conditioned fusion head of PhyChronos-Irr (gate w, action gain g, spread s),
trained with a NEW objective on paired factual / counterfactual action sequences:

  L = L_pinball(factual)                                            (accuracy on what happened)
    + l_mono  * E_k[ relu( m(u_k) - m(u_{k+1}) ) ]                  (dose monotonicity)
    + l_band  * E_k[ relu( |r_k - e_k| - delta*|e_k| - eps ) ]       (physical response band)
    + l_caus  * E_k[ |r_k|  on hours before the dose starts ]        (no response before the action)
    + l_sat   * E_k[ relu( q_k - SP ) + relu( PWP_low - q_k ) ]      (stay inside the soil-water range)
  with r_k = m(u_k) - m(0) the predicted counterfactual response and e_k = p(u_k) - p(0) the physics
  response, for counterfactual doses u_k in {30, 60, 120, 240} min starting at hour 2.
Reliability: the head sees 7 sensor-reliability features and is trained on clean + fault-augmented
contexts (E6b phase groups, clean labels), so trust in the sensor/state is learned jointly with the response.
Outputs: E2 predictions "RACER-Irr"; dose-response diagnostics results/e12/racer_dose.csv.
"""
import sys
import numpy as np
import pandas as pd
import torch
import config as C
import e2_panel as P
import physics as PH
import e3_hybrid as HY
from e3_physics import hourly_with_et0
from e6b_fault_aug import phase_groups, build_cache, FAULTS
from e6_faults import perturb

H = P.H
DOSES = [0, 30, 60, 120, 240]
DOSE_H = 2
LAM = dict(mono=1.0, band=1.0, caus=1.0, sat=0.2)
DELTA, EPS = 0.5, 0.1
OUT = C.RESULTS / "e12"


def cf_set(z, h, keep, bucket, x):
    """Counterfactual physics paths and features for each dose: lists of (x_k, p_k) with p_k [B,H,2]."""
    _, th0, I, E, _ = PH.arrays(h, keep)
    med = h["sm_f"].rolling(3, min_periods=1).median()
    th0 = torch.tensor(med.reindex(keep).values, dtype=torch.float32)
    flow = C.THRESH[z]["flow"]
    I0 = I.clone(); I0[:, PH.NPAST:] = 0
    with torch.no_grad():
        p0 = bucket(th0, I0, E)
    out = []
    for d in DOSES:
        Ik = I0.clone()
        rem, j = d, DOSE_H - 1
        while rem > 0 and j < H:
            m = min(60, rem); Ik[:, PH.NPAST + j] = m * flow; rem -= m; j += 1
        with torch.no_grad():
            pk = bucket(th0, Ik, E)
        xk = x.clone()
        xk[:, 3] = Ik[:, PH.NPAST:].sum(1) / 50
        xk[:, 5] = (pk - p0).mean(1) / 10
        out.append((xk, torch.stack([pk, p0], -1)))
    return out


def racer_loss(head, x, c, p, y, cfs, z, use_cf=True):
    q, _ = head(x, c, p)
    L = HY.pinball(q, y)
    if not use_cf:
        return L
    t_ = C.THRESH[z]
    ms, qs = [], []
    for xk, pk in cfs:
        qk, _ = head(xk, c, pk)
        ms.append(qk[..., 4]); qs.append(qk)
    m0 = ms[0]
    mono = sum(torch.relu(ms[k] - ms[k + 1]).mean() for k in range(len(ms) - 1))
    band, caus, sat = 0.0, 0.0, 0.0
    for k in range(1, len(ms)):
        r = ms[k] - m0
        e = cfs[k][1][..., 0] - cfs[k][1][..., 1]
        band = band + torch.relu((r - e).abs() - DELTA * e.abs() - EPS).mean()
        caus = caus + r[:, :DOSE_H - 1].abs().mean()
    for qk in qs:
        sat = sat + (torch.relu(qk - t_["SP"]) + torch.relu(t_["PWP"] - 10 - qk)).mean()
    return L + LAM["mono"] * mono + LAM["band"] * band + LAM["caus"] * caus + LAM["sat"] * sat


def train_racer(X, Cq, Pp, Y, CF, z, use_cf=True):
    torch.manual_seed(C.SEED)
    head = HY.Head(X.shape[1], True)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3, weight_decay=1e-4)
    n = len(X); cut = int(n * 0.85)
    tr = lambda a: a[:cut]
    va = lambda a: a[cut:]
    best, state, bad = 1e9, None, 0
    for ep in range(400):
        head.train()
        loss = racer_loss(head, tr(X), tr(Cq), tr(Pp), tr(Y), [(tr(a), tr(b)) for a, b in CF], z, use_cf)
        opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vl = racer_loss(head, va(X), va(Cq), va(Pp), va(Y), [(va(a), va(b)) for a, b in CF], z, use_cf).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad > 30:
                break
    head.load_state_dict(state); head.eval()
    return head


def assemble(z, h0, tr, cdf, aug, bucket, lim, fault_aug=True):
    parts = []
    k, x, c, p, y = HY.build_xy(z, h0, tr, cdf, bucket, True)
    if k:
        parts.append((x, c, p, y, cf_set(z, h0, k, bucket, x)))
    if fault_aug:
        for ph, ts in phase_groups(z, h0).items():
            ts = [t for t in ts if t <= lim]
            if not ts:
                continue
            hp = perturb(h0, ts, FAULTS[ph % 4])
            cp = aug[(aug.sector == z) & (aug.phase == ph)]
            k, x, c, p, _ = HY.build_xy(z, hp, ts, cp, bucket, True)
            if not k:
                continue
            y = PH.arrays(h0, k)[4]
            parts.append((x, c, p, y, cf_set(z, hp, k, bucket, x)))
    cat = lambda i: torch.cat([q[i] for q in parts])
    CF = [(torch.cat([q[4][d][0] for q in parts]), torch.cat([q[4][d][1] for q in parts])) for d in range(len(DOSES))]
    X, Cq, Pp, Y = cat(0), cat(1), cat(2), cat(3)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(C.SEED))
    return X[perm], Cq[perm], Pp[perm], Y[perm], [(a[perm], b[perm]) for a, b in CF]


def dose_diag(head, z, h, keep, cdf, bucket, name):
    k, x, c, p, _ = HY.build_xy(z, h, keep, cdf, bucket, True)
    cfs = cf_set(z, h, k, bucket, x)
    with torch.no_grad():
        qs_all = [head(xk, c, pk)[0].numpy() for xk, pk in cfs]
        ms = [q[..., 4] for q in qs_all]
        es = [(pk[..., 0] - pk[..., 1]).numpy() for _, pk in cfs]
    t_ = C.THRESH[z]
    post = slice(DOSE_H - 1, None)
    resp_err = float(np.mean([np.abs((ms[i] - ms[0])[:, post] - es[i][:, post]).mean() for i in range(1, len(DOSES))]))
    band_viol = float(np.mean([(np.abs((ms[i] - ms[0])[:, post] - es[i][:, post]) > DELTA * np.abs(es[i][:, post]) + EPS).mean()
                               for i in range(1, len(DOSES))]))
    sat_viol = float(np.mean([((q > t_["SP"]) | (q < t_["PWP"] - 10)).mean() for q in qs_all]))
    rows = []
    r3 = np.stack([ms[i][:, DOSE_H] - ms[0][:, DOSE_H] for i in range(len(DOSES))], 1)
    mono = float((np.diff(r3, axis=1) >= -0.05).all(1).mean())
    pre = float(np.mean([np.abs(ms[i][:, :DOSE_H - 1] - ms[0][:, :DOSE_H - 1]).mean() for i in range(1, len(DOSES))]))
    for i, d in enumerate(DOSES):
        rows.append(dict(model=name, sector=z, dose=d, resp_3h=float(r3[:, i].mean()),
                         peak=float((ms[i] - ms[0]).max(1).mean()), phys_peak=float(es[i].max(1).mean()),
                         monotone_frac=mono, pre_dose_leak=pre, resp_err=resp_err, band_viol=band_viol, sat_viol=sat_viol))
    return rows


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "full"      # full | nocf | nomono | noband | nocaus | nosat | nofa
    names = {"full": "RACER-Irr (rerun)", "nocf": "RACER-Irr (no CF loss)", "nomono": "RACER-Irr (no monotonicity)",
             "noband": "RACER-Irr (no physics band)", "nocaus": "RACER-Irr (no causality)", "nosat": "RACER-Irr (no saturation)",
             "nofa": "RACER-Irr (no fault augmentation)"}
    name = names[variant]
    use_cf = variant != "nocf"
    fault_aug = variant != "nofa"
    for key, v in (("nomono", "mono"), ("noband", "band"), ("nocaus", "caus"), ("nosat", "sat")):
        if variant == key:
            LAM[v] = 0.0
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    cdf = HY.chronos_all()
    aug = build_cache()
    frames, diag = [], []
    for z, f, _, orig in P.all_tasks():
        if not orig:
            continue
        h0 = hourly_with_et0(z)
        bucket = HY.load_bucket(z, f["fold"])
        lim = f["train_end"] - pd.Timedelta(hours=H)
        avail = set(cdf.loc[cdf.sector == z, "origin"])
        tr = [t for t in HY.season_origins(z, h0) if t <= lim and t in avail]
        X, Cq, Pp, Y, CF = assemble(z, h0, tr, cdf, aug, bucket, lim, fault_aug)
        head = train_racer(X, Cq, Pp, Y, CF, z, use_cf)
        te = [t for t in orig if t in avail]
        k, x, c, p, _ = HY.build_xy(z, h0, te, cdf, bucket, True)
        with torch.no_grad():
            q, _ = head(x, c, p)
        for t, qq in zip(k, q.numpy()):
            frames.append(P.pred_frame(z, f["fold"], t, qq[:, 4], qq))
        if f["fold"] == 2:
            fut_w = h0["liters"].fillna(0)[::-1].rolling(H, min_periods=1).sum()[::-1].shift(-1)
            dts = [t for t in te if fut_w.get(t, 1) == 0][:60] or te[:60]
            diag += dose_diag(head, z, h0, dts, cdf, bucket, name)
        print(f"S{z} f{f['fold']} done (train {len(X)})", flush=True)
    P.save_preds(name, frames)
    D = pd.DataFrame(diag)
    fn = OUT / "racer_dose.csv"
    if fn.exists():
        old = pd.read_csv(fn); D = pd.concat([old[old.model != name], D])
    D.to_csv(fn, index=False)
    print(D[D.model == name].pivot_table(index="sector", columns="dose", values="peak").round(2).to_string())
    print(D[D.model == name].groupby("sector")[["monotone_frac", "pre_dose_leak"]].first().round(3).to_string())


if __name__ == "__main__":
    main()
