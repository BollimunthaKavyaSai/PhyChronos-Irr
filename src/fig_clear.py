"""Redrawn Figure 6 (calibration) and Figure 8 (water-stress frontier) with non-overlapping labels."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from figures import S, INK, MUTED, save

# ---------------- Figure 6: calibration (two stacked panels, labels rotated, legend outside)
r = pd.read_csv(C.RESULTS / "e7" / "e7_results.csv")
models = ["XGBoost-Q", "Chronos-2", "TimesFM-2.5", "TiRex", "PhyChronos-Irr-FA"]
methods = ["Uncalibrated", "Split-CQR", "Rolling-CQR", "ACI-QT"]
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4))
x = np.arange(len(models)); w = 0.2
for ax, col, lab, fmt in zip(axes, ["coverage", "width"], ["Coverage of the 80 % interval", "Mean interval width (points)"], ["{:.2f}", "{:.1f}"]):
    top = 0
    for i, me in enumerate(methods):
        v = [float(r[(r.model == m) & (r.method == me)][col].iloc[0]) for m in models]
        b = ax.bar(x + (i - 1.5) * w, v, w * 0.9, color=S[i], label=me)
        for rect, val in zip(b, v):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + (0.004 if col == "coverage" else 0.15), fmt.format(val),
                    ha="center", va="bottom", fontsize=6.5, rotation=90, color=INK)
        top = max(top, max(v))
    ax.set_xticks(x, models, fontsize=8)
    ax.set_ylabel(lab)
    if col == "coverage":
        ax.axhline(0.8, color=INK, lw=1, ls="--")
        ax.text(-0.55, 0.806, "target 0.80", fontsize=7, color=INK, ha="left", va="bottom")
        ax.set_ylim(0.6, 1.0)
        ax.set_title("(a) Coverage", loc="left", fontsize=9, color=INK)
    else:
        ax.set_ylim(0, top * 1.25)
        ax.set_title("(b) Interval width", loc="left", fontsize=9, color=INK)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig_e7_conformal")

# ---------------- Figure 8: (a) full view with controller names, (b) zoom with risk-margin labels
T = pd.read_csv(C.RESULTS / "e8" / "e8_pareto.csv")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 8.2), gridspec_kw=dict(height_ratios=[1, 1.15]))
sty = {"Physics": (S[3], "o", "Physics planner"), "Chronos-BPT": (S[0], "s", "BPT"), "RACER-Irr": (S[1], "^", "RACER-Irr")}
refs = [("Hysteresis", INK, "Farm hysteresis", (-12, 10), "right"), ("EDML", S[2], "EDML", (10, -4), "left"),
        ("Chronos-2", S[4], "Chronos-2 planner", (10, -14), "left"), ("Fixed-interval", MUTED, "Fixed interval", (10, -4), "left")]
for ax in (ax1, ax2):
    for c, (col, mk, name) in sty.items():
        g = T[T.controller == c].sort_values("water_L")
        ax.plot(g.water_L / 1000, g.stress_h, "-", marker=mk, color=col, lw=1.8, ms=6, label=name)
    for c, col, name, off, ha in refs:
        g = T[T.controller == c]
        if len(g):
            ax.scatter(g.water_L / 1000, g.stress_h, marker="D", s=40, color=col, zorder=3,
                       label=name if ax is ax1 else None)
    ax.set_ylabel("Hours below MADP (crop stress)")
for c, col, name, off, ha in refs:
    g = T[T.controller == c]
    ax1.annotate(name, (g.water_L.iloc[0] / 1000, g.stress_h.iloc[0]), xytext=off, textcoords="offset points", fontsize=8, color=col, ha=ha)
ax1.set_xlim(0, 60); ax1.set_ylim(-200, 5000)
ax1.add_patch(plt.Rectangle((31, -80), 25, 1180, fill=False, ls="--", lw=0.9, color=MUTED))
ax1.text(31.3, 1180, "zoomed in (b)", fontsize=7.5, color=MUTED)
ax1.set_title("(a) All controllers", loc="left", fontsize=9, color=INK)
ax1.legend(fontsize=7.5, loc="upper right", ncol=2)
lab = {("Physics", -6.0): (6, 12, "left"), ("Physics", -4.5): (0, -18, "center"), ("Physics", -3.0): (0, -18, "center"),
       ("Physics", -1.5): (0, -18, "center"), ("Physics", 0.0): (0, -18, "center"),
       ("Chronos-BPT", 0.0): (8, 6, "left"), ("Chronos-BPT", 1.5): (-10, -4, "right"), ("Chronos-BPT", 3.0): (-10, -12, "right"),
       ("RACER-Irr", 0.0): (8, 0, "left"), ("RACER-Irr", 1.5): (-10, -6, "right"), ("RACER-Irr", 3.0): (8, -20, "left")}
for (c, k), (dx, dy, ha) in lab.items():
    row = T[(T.controller == c) & (T.kappa == k)].iloc[0]
    ax2.annotate(f"κ = {k:+g}", (row.water_L / 1000, row.stress_h), xytext=(dx, dy), textcoords="offset points",
                 fontsize=7.5, color=sty[c][0], ha=ha)
ax2.set_xlim(31, 56); ax2.set_ylim(-110, 1100)
ax2.set_xlabel("Total water applied, sectors 1, 2, 4 and 5 (m³)")
ax2.set_title("(b) Zoom: risk-margin sweep (κ = safety margin added to MADP)", loc="left", fontsize=9, color=INK)
ax2.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
save(fig, "fig_e8_pareto")
