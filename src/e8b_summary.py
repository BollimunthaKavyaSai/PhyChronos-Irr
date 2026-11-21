"""Merge E8b (forecaster-in-the-loop), BPT and E8 reference controllers; totals per plant; figure."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from figures import S, INK, MUTED, save

suf = "_noise" if "--noise" in sys.argv else ""
D = C.RESULTS / "e8"
parts = [D / f"e8b_results{suf}.csv", D / f"e8b_results{suf}_bpt.csv", D / f"e8_refs{suf}.csv"]
R = pd.concat([pd.read_csv(p) for p in parts if p.exists()], ignore_index=True)
R = R.drop_duplicates(["sector", "plant", "controller"], keep="last")
R.to_csv(D / f"e8b_all{suf}.csv", index=False)
ORDER = ["Fixed-interval", "Hysteresis", "EDML", "Physics", "Chronos-2", "PhyChronos-Irr", "PhyChronos-Delta", "Chronos-BPT", "RACER-Irr"]
per = R.pivot_table(index=["plant", "controller"], columns="sector", values=["water_L", "stress_h"]).round(0)
tot = R.groupby(["plant", "controller"]).agg(water_L=("water_L", "sum"), stress_h=("stress_h", "sum"),
                                             danger_h=("danger_h", "sum"), excess_h=("excess_h", "sum"),
                                             optimal=("optimal_frac", "mean"), n=("sector", "count")).reset_index()
ref = tot[tot.controller == "Hysteresis"].set_index("plant")["water_L"]
tot["water_vs_farm_%"] = [100 * (w / ref[p] - 1) for p, w in zip(tot.plant, tot.water_L)]
tot["controller"] = pd.Categorical(tot.controller, ORDER, ordered=True)
tot = tot.sort_values(["plant", "controller"])
tot.to_csv(D / f"e8b_totals{suf}.csv", index=False)
print(tot.round(2).to_string(index=False))
print(per.to_string())

OFF = {"Fixed-interval": (6, -2), "Hysteresis": (-30, 8), "EDML": (6, 6), "Physics": (-40, 8), "Chronos-2": (6, -12),
       "PhyChronos-Irr": (6, 4), "PhyChronos-Delta": (6, 4), "Chronos-BPT": (6, -10), "RACER-Irr": (-62, -12)}
fig, ax = plt.subplots(figsize=(6.2, 3.6))
g = tot[tot.plant == "A"]
for i, c in enumerate(ORDER):
    row = g[g.controller == c]
    if len(row):
        ax.scatter(row.water_L / 1000, row.stress_h, s=55, color=(MUTED if c == "Fixed-interval" else "#111111" if c == "RACER-Irr" else S[i % len(S)]), edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(c, (row.water_L.iloc[0] / 1000, row.stress_h.iloc[0]), xytext=OFF.get(c, (4, 3)), textcoords="offset points", fontsize=7, color=INK)
ax.set_xlabel("Total water, 4 sectors (m³)")
ax.set_ylabel("Hours below MADP (stress)")
ax.set_title("Same planner, different forecasters (physics plant; lower-left is better)" + (" - noisy" if suf else ""), loc="left", fontsize=9.5, color=INK)
save(fig, f"fig_e8b_control{suf}")
