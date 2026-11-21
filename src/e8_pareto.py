"""Water-stress frontier on plant A from the risk-margin sweep (kappa) for Physics, Chronos-BPT, RACER-Irr."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from figures import S, INK, MUTED, save

D = C.RESULTS / "e8"
rows = []
for f, k in [("e8b_results.csv", 0.0), ("e8b_results_bpt.csv", 0.0), ("e8b_results_k1.5.csv", 1.5), ("e8b_results_k3.csv", 3.0),
             ("e8b_results_k-1.5.csv", -1.5), ("e8b_results_k-3.csv", -3.0), ("e8b_results_k-4.5.csv", -4.5), ("e8b_results_k-6.csv", -6.0), ("e8_refs.csv", 0.0)]:
    p = D / f
    if p.exists():
        d = pd.read_csv(p)
        d["kappa"] = k
        rows.append(d[d.plant == "A"])
R = pd.concat(rows).drop_duplicates(["sector", "controller", "kappa"], keep="last")
T = R.groupby(["controller", "kappa"]).agg(water_L=("water_L", "sum"), stress_h=("stress_h", "sum"),
                                           optimal=("optimal_frac", "mean"), n=("sector", "count")).reset_index()
T = T[T.n == 4]
farm = T[T.controller == "Hysteresis"].iloc[0]
T["water_vs_farm_%"] = 100 * (T.water_L / farm.water_L - 1)
T["stress_vs_farm_%"] = 100 * (T.stress_h / farm.stress_h - 1)
T.to_csv(D / "e8_pareto.csv", index=False)
print(T.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.2, 3.6))
styles = {"Physics": S[3], "Chronos-BPT": S[0], "RACER-Irr": S[1]}
for c, col in styles.items():
    g = T[T.controller == c].sort_values("water_L")
    ax.plot(g.water_L / 1000, g.stress_h, "-o", color=col, lw=2, ms=5, label=c)
    for _, r in g.iterrows():
        ax.annotate(f"κ={r.kappa:+g}", (r.water_L / 1000, r.stress_h), xytext=(4, 4), textcoords="offset points", fontsize=6.5, color=MUTED)
for c, col in [("Hysteresis", INK), ("EDML", S[2]), ("Chronos-2", S[4]), ("Fixed-interval", MUTED)]:
    g = T[T.controller == c]
    if len(g):
        ax.scatter(g.water_L / 1000, g.stress_h, marker="s", s=40, color=col, zorder=3, label=c + (" (farm)" if c == "Hysteresis" else ""))
ax.set_xlabel("Total water, 4 sectors (m³)")
ax.set_ylabel("Hours below MADP (stress)")
ax.legend(fontsize=7, ncol=2)
ax.set_title("Water-stress frontier on the physics plant (lower-left is better)", loc="left", fontsize=10, color=INK)
save(fig, "fig_e8_pareto")
