"""New figures for the reviewer-oriented revision: evaluation timeline, identifiability, dose-response curves."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config as C
import e2_panel as P
from figures import S, INK, MUTED, save

# ---------------- evaluation timeline (actual fold dates)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
crop = {1: "tomato", 2: "tomato", 4: "zucchini", 5: "blueberry"}
for row, z in enumerate(C.PRIMARY_SECTORS[::-1]):
    h = P.hourly(z)
    a, b = P.season_window(z)
    fl = P.folds(z, h)
    y = row
    first_test = fl[0]["test_start"]
    ax.barh(y, mdates.date2num(fl[0]["train_end"]) - mdates.date2num(a), left=mdates.date2num(a), height=0.5, color=S[0], alpha=0.35,
            label="training period of fold 0 (expands with each fold)" if row == 0 else None)
    for f in fl:
        ax.barh(y, mdates.date2num(f["test_start"]) - mdates.date2num(f["train_end"]), left=mdates.date2num(f["train_end"]), height=0.5,
                color=INK, label="24-h embargo" if (row == 0 and f["fold"] == 0) else None)
        ax.barh(y, mdates.date2num(f["test_end"]) - mdates.date2num(f["test_start"]), left=mdates.date2num(f["test_start"]), height=0.5,
                color=S[1 + f["fold"] % 2], edgecolor="white", linewidth=0.8,
                label=("test fold (alternating colours)" if (row == 0 and f["fold"] == 0) else None))
        ax.text(mdates.date2num(f["test_start"]) + (mdates.date2num(f["test_end"]) - mdates.date2num(f["test_start"])) / 2, y,
                f"F{f['fold']}", ha="center", va="center", fontsize=6.5, color="white")
ax.set_yticks(range(4), [f"Sector {z} ({crop[z]})" for z in C.PRIMARY_SECTORS[::-1]], fontsize=7.5)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_xlim(mdates.date2num(pd.Timestamp("2025-02-10")), mdates.date2num(pd.Timestamp("2025-09-08")))
ax.set_xlabel("2025 (irrigation season of each sector)")
ax.legend(fontsize=7, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.16), frameon=False)
ax.grid(axis="y", visible=False)
fig.subplots_adjust(left=0.22)
save(fig, "fig_protocol_timeline")

# ---------------- identifiability (profile-likelihood MAE range, log scale)
idf = pd.read_csv(C.RESULTS / "e3" / "e4_identifiability.csv")
params = ["alpha", "c", "kappa", "d", "r", "beta"]
NL = chr(10)
lab = {"alpha": "α" + NL + "(water input)", "c": "c" + NL + "(ET demand)", "kappa": "κ" + NL + "(probe thermal)",
       "d": "d" + NL + "(drainage)", "r": "r" + NL + "(redistribution)", "beta": "β" + NL + "(retained fraction)"}
fig, ax = plt.subplots(figsize=(7.2, 3.2))
w = 0.2
for i, z in enumerate(C.PRIMARY_SECTORS):
    v = [max(float(idf[(idf.sector == z) & (idf.param == p)]["range_%"].iloc[0]), 0.05) for p in params]
    ax.bar(np.arange(len(params)) + (i - 1.5) * w, v, w * 0.9, color=S[i], label=f"Sector {z}")
ax.axhline(2, color=INK, ls="--", lw=1)
ax.text(5.45, 2.3, "identifiability threshold (2 %)", ha="right", fontsize=7, color=INK)
ax.set_yscale("log")
ax.set_xticks(range(len(params)), [lab[p] for p in params], fontsize=6.8)
ax.set_ylabel("Training-MAE range across\nparameter bounds (%)")
ax.legend(fontsize=7, ncol=4, loc="upper right")
save(fig, "fig_identifiability")

# ---------------- dose-response curves
e3c = pd.read_csv(C.RESULTS / "e3" / "e3c_dose_response.csv")
rd = pd.read_csv(C.RESULTS / "e12" / "racer_dose.csv")
rd = rd[rd.model == "RACER-Irr"]
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=False)
for ax, z in zip(axes, C.PRIMARY_SECTORS):
    for m, col, name in [("chronos", S[4], "Chronos-2+cov"), ("hybrid", S[3], "PhyChronos-Irr")]:
        g = e3c[(e3c.sector == z) & (e3c.model == m)].sort_values("dose")
        ax.plot(g.dose, g.peak, "-o", ms=3, color=col, lw=1.5, label=name)
    g = rd[rd.sector == z].sort_values("dose")
    ax.plot(g.dose, g.peak, "-s", ms=3, color=S[1], lw=1.8, label="RACER-Irr")
    ax.plot(g.dose, g.phys_peak, "--", color=S[2], lw=1.2, label="physics model")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_title(f"Sector {z}", fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("planned dose (min)", fontsize=7.5)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("peak forecast response (points)", fontsize=7.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "fig_dose_response")

# ---------------- independent plant: water-stress frontier under the risk-margin sweep
E8 = C.RESULTS / "e8"
def tot(fn, ctrl):
    d = pd.read_csv(E8 / fn)
    d = d[d.controller == ctrl]
    return d.water_L.sum() / 1000, d.stress_h.sum()
base = pd.read_csv(E8 / "e8b_results_plantloso.csv").groupby("controller")[["water_L", "stress_h"]].sum()
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for ctrl, name in [("Hysteresis", "farm hysteresis"), ("Fixed-interval", "fixed interval"), ("EDML", "EDML"), ("Chronos-2", "Chronos-2 planner")]:
    w, s = base.loc[ctrl, "water_L"] / 1000, base.loc[ctrl, "stress_h"]
    ax.scatter(w, s, marker="D", s=34, color=MUTED if ctrl != "Hysteresis" else INK, zorder=3)
    off = {"Fixed-interval": (-12, 8), "Chronos-2": (7, -10), "EDML": (6, 2), "Hysteresis": (-30, 8)}[ctrl]
    ax.annotate(name, (w, s), xytext=off, textcoords="offset points", fontsize=7, color=INK)
series = [("Physics", S[2], [("e8b_results_plantloso_phys_k-6.csv", "−6"), ("e8b_results_plantloso_phys_k-3.csv", "−3"), ("e8b_results_plantloso.csv", "0")]),
          ("Chronos-BPT", S[0], [("e8b_results_plantloso.csv", "0"), ("e8b_results_plantloso_k1.5.csv", "+1.5"), ("e8b_results_plantloso_k3.csv", "+3")]),
          ("RACER-Irr", S[1], [("e8b_results_plantloso.csv", "0"), ("e8b_results_plantloso_racer_k3.csv", "+3")]),
          ("Ensemble-BPT", S[3], [("e8b_results_plantloso_ensbpt.csv", "0"), ("e8b_results_plantloso_ensbpt_k3.csv", "+3")])]
lab = {"Physics": "physics planner", "Chronos-BPT": "BPT (proposed)", "RACER-Irr": "RACER-Irr (proposed)",
       "Ensemble-BPT": "Ensemble-BPT (proposed)"}
for ctrl, col, pts in series:
    xy = [tot(f, ctrl) for f, _ in pts]
    ax.plot([p[0] for p in xy], [p[1] for p in xy], "-o", color=col, lw=1.8, ms=5, label=lab[ctrl], zorder=4)
    for (w, s), (_, k) in zip(xy, pts):
        off = {("Physics", "−6"): (6, 4), ("Physics", "−3"): (-6, 8), ("Physics", "0"): (-10, 8), ("Chronos-BPT", "0"): (6, 2),
               ("Chronos-BPT", "+1.5"): (-38, -8), ("Chronos-BPT", "+3"): (-34, -4), ("RACER-Irr", "0"): (6, 0), ("RACER-Irr", "+3"): (-36, -4),
               ("Ensemble-BPT", "0"): (6, 6), ("Ensemble-BPT", "+3"): (-64, 10)}[(ctrl, k)]
        ax.annotate(f"κ = {k}", (w, s), xytext=off, textcoords="offset points", fontsize=6.5, color=col)
ax.set_ylim(0, 4500)
ax.set_xlabel("total water applied (m³, four sectors)")
ax.set_ylabel("stress hours (θ < MADP)")
ax.legend(fontsize=7.5, loc="upper right", frameon=False)
save(fig, "fig_indep_frontier")
