"""Architecture figure for the manuscript (300 dpi PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import config as C

FIG = C.ROOT / "paper" / "figures"
INK = "#1f1f1e"
COL = {"data": ("#f1efe8", "#5f5e5a"), "fm": ("#eeedfe", "#534ab7"), "phys": ("#e1f5ee", "#0f6e56"),
       "dec": ("#faece7", "#993c1d"), "new": ("#fff4d6", "#9a6a00")}

fig, ax = plt.subplots(figsize=(7.2, 6.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, sub, kind):
    fc, ec = COL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=8.2, color=ec, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.6, color=INK)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#6b6a64", lw=0.9, shrinkA=0, shrinkB=0))


box(4, 90, 42, 8, "Sensor telemetry", "soil moisture, valve, flow, EC", "data")
box(54, 90, 42, 8, "Weather", "ERA5, FAO-56 ET0, radiation", "data")
arrow(25, 90, 25, 85); arrow(75, 90, 75, 85)
box(4, 76, 92, 8, "Data audit and quality control (C1)", "true-time targets, deadband carry-forward, fault flags, in-season rolling-origin protocol", "data")
arrow(17, 76, 17, 70); arrow(50, 76, 50, 70); arrow(83, 76, 83, 70)
box(2, 60, 30, 9, "Chronos-2 (+covariates)", "frozen, zero-shot, 9 quantiles", "fm")
box(35, 60, 30, 9, "Reliability features", "7 sensor-fault signals", "data")
box(68, 60, 30, 9, "Dual-store physics (C2)", "paths p(u), p(0), p(π_b)", "phys")
arrow(17, 60, 30, 53); arrow(50, 60, 50, 53); arrow(83, 60, 70, 53)
box(14, 41, 72, 11, "Fusion head: PhyChronos-Irr (C2) / RACER-Irr (C4)",
    "m = w(c_med + g·[p(u)−p(0)]) + (1−w)·p(u);  RACER: counterfactual doses,\nmonotonicity, causality, physical band, fault augmentation", "new")
box(2, 26, 46, 11, "Behaviour-Policy Transport (C4)", "training-free:  q(u) = q_FM − p(π_b) + p(u)", "new")
box(52, 26, 46, 11, "Conformal calibration", "online ACI quantile tracking, 80 % intervals", "fm")
arrow(40, 41, 25, 37.5); arrow(60, 41, 75, 37.5)
ax.annotate("", xy=(25, 37.5), xytext=(17, 60), arrowprops=dict(arrowstyle="-|>", color="#6b6a64", lw=0.9, ls="--", connectionstyle="arc3,rad=0.25"))
box(14, 11, 72, 10, "Chance-constrained planner (C3)", "daily dose: least water s.t. P(θ ≥ MADP) ≥ 0.9 (+ risk margin κ)", "dec")
arrow(25, 26, 40, 21.5); arrow(75, 26, 60, 21.5)
box(22, 0.5, 56, 7, "Greenhouse sector (simulated plant)", "irrigation applied; new readings", "data")
arrow(50, 11, 50, 8)
fig.savefig(FIG / "fig_architecture.png", dpi=300, bbox_inches="tight")
print("saved")
