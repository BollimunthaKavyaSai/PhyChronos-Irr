"""Publication figures (300 dpi PNG) from results/. Run: python figures.py [names...]"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

FIG = C.ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def bars(ax, groups, series, values, colors, fmt="{:.2f}", label_last=False):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        b = ax.bar(x + (i - (n - 1) / 2) * w, values[i], w * 0.92, color=colors[i], label=s)
        for rect, v in zip(b, values[i]):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, groups)


def fig_e1():
    r = pd.read_csv(C.RESULTS / "e1" / "e1_summary_primary_full.csv")
    s = pd.read_csv(C.RESULTS / "e1" / "e1_summary_primary_season.csv")
    get = lambda d, data, prot, model: float(d[(d.data == data) & (d.protocol == prot) & (d.model == model)]["MAE"].iloc[0])
    groups = ["Shuffled 3-day CV\n(published protocol)", "Rolling-origin\nfull period", "Rolling-origin\nin-season (ours)"]
    pers = [get(r, "published", "SGKF-3day", "Persistence"), get(r, "published", "Rolling+embargo", "Persistence"),
            get(s, "rebuilt", "Rolling+embargo", "Persistence")]
    xgb = [get(r, "published", "SGKF-3day", "XGBoost"), get(r, "published", "Rolling+embargo", "XGBoost"),
           get(s, "rebuilt", "Rolling+embargo", "XGBoost-delta")]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bars(ax, groups, ["Persistence", "XGBoost (EDML-style)"], [pers, xgb], [S[0], S[1]])
    ax.set_ylabel("MAE of 24-h mean soil moisture (%)")
    ax.legend(loc="upper left")
    ax.set_title("Evaluation protocol changes the conclusion", loc="left", fontsize=10, color=INK)
    save(fig, "fig_e1_protocol")


def fig_e2():
    import e2_panel as P
    files = ["Persistence", "XGBoost-Q", "DLinear", "PatchTST", "TimesFM-2.5", "Moirai-2.0", "Chronos-2", "PhyChronos-Irr"]
    truth = {}
    for z in C.PRIMARY_SECTORS:
        truth[z] = P.hourly(z)["sm"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    k = 0
    for name in files:
        f = P.PRED_DIR / f"{name}.csv.gz"
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["origin"])
        d["y"] = [truth[z].get(o + pd.Timedelta(hours=int(h))) for z, o, h in zip(d.sector, d.origin, d.h)]
        d = d.dropna(subset=["y"])
        m = (d.point - d.y).abs().groupby(d.h).mean()
        ax.plot(m.index, m.values, lw=2, color=S[k], label=name, marker="o" if name == "PhyChronos-Irr" else None, ms=3)
        k += 1
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("MAE (soil-moisture points)")
    ax.legend(ncol=2, fontsize=7.5)
    ax.set_title("Hourly soil-moisture forecast error by horizon (sectors 1, 2, 4, 5)", loc="left", fontsize=10, color=INK)
    save(fig, "fig_e2_horizon")


def fig_e3b():
    r = pd.read_csv(C.RESULTS / "e3" / "e3b_action_sensitivity.csv")
    groups = [f"Sector {z}" for z in r.sector]
    vals = [r.empirical_rise_3h.values, r.chronos_peak_response.values, r.phys_peak_response.values, r.hybrid_peak_response.values]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bars(ax, groups, ["Observed (diff-in-diff)", "Chronos-2+cov", "Dual-store physics", "PhyChronos-Irr"], vals,
         [MUTED, S[0], S[2], S[1]], fmt="{:.1f}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Response to a 30-min dose (points)")
    ax.legend(ncol=2, fontsize=7.5, loc="upper left")
    ax.set_title("Action sensitivity: does the forecast react to planned irrigation?", loc="left", fontsize=10, color=INK)
    save(fig, "fig_e3b_action")


def fig_e7():
    r = pd.read_csv(C.RESULTS / "e7" / "e7_results.csv")
    models = ["XGBoost-Q", "Chronos-2", "TimesFM-2.5", "TiRex", "PhyChronos-Irr-FA"]
    methods = ["Uncalibrated", "Split-CQR", "Rolling-CQR", "ACI-QT"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, col, lab in zip(axes, ["coverage", "width"], ["Empirical coverage of 80 % interval", "Mean interval width (points)"]):
        vals = [[float(r[(r.model == m) & (r.method == me)][col].iloc[0]) for m in models] for me in methods]
        bars(ax, models, methods, vals, S[:4], fmt="{:.2f}" if col == "coverage" else "{:.1f}")
        if col == "coverage":
            ax.axhline(0.8, color=INK, lw=1, ls="--")
            ax.set_ylim(0.5, 0.95)
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(ncol=2, fontsize=7, loc="lower right")
    fig.suptitle("Conformal calibration under temporal shift (folds 1-3)", x=0.01, ha="left", fontsize=10, color=INK)
    save(fig, "fig_e7_conformal")


def fig_e6():
    t = pd.read_csv(C.RESULTS / "e6" / "e6_summary.csv", index_col=0)
    scen = ["clean", "dropout", "stuck", "drift", "spike"]
    models = [m for m in ["Persistence", "GH-DualStore", "Chronos-2+cov", "PhyChronos-Irr (no fault feat.)", "PhyChronos-Irr", "PhyChronos-Irr-FA"] if m in t.index]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    bars(ax, scen, models, [t.loc[m, scen].values for m in models], [MUTED, S[2], S[0], S[3], S[4], S[1]][:len(models)], fmt="{:.1f}")
    ax.set_ylabel("MAE 1-24 h (points)")
    ax.legend(ncol=2, fontsize=7)
    ax.set_title("Robustness to sensor faults at forecast time", loc="left", fontsize=10, color=INK)
    save(fig, "fig_e6_faults")


def fig_e8():
    for suffix in ["", "_noise"]:
        f = C.RESULTS / "e8" / f"e8_results{suffix}.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        g = r.groupby(["plant", "controller"])[["water_L", "stress_h"]].sum().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
        ctrls = ["Fixed-interval", "Hysteresis", "Predictive-threshold", "EDML", "D-MPC", "CC-MPC"]
        for ax, plant, title in zip(axes, ["A", "B"], ["Plant A: physics simulator", "Plant B: data-driven simulator"]):
            for i, c in enumerate(ctrls):
                row = g[(g.plant == plant) & (g.controller == c)]
                if len(row):
                    ax.scatter(row.water_L, row.stress_h, s=60, color=S[i], edgecolor="white", linewidth=1.5, label=c, zorder=3)
                    ax.annotate(c, (row.water_L.iloc[0], row.stress_h.iloc[0]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK)
            ax.set_title(title, fontsize=9, color=INK, loc="left")
            ax.set_xlabel("Total water applied (L)")
        axes[0].set_ylabel("Hours below MADP (stress)")
        fig.suptitle("Closed-loop irrigation: water vs. crop stress (lower-left is better)" + (" - ET0 forecast error + process noise" if suffix else ""),
                     x=0.01, ha="left", fontsize=10, color=INK)
        save(fig, f"fig_e8_control{suffix}")


def fig_e10():
    f = C.RESULTS / "e10_data_efficiency.csv"
    if not f.exists():
        return
    r = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, m in enumerate(["XGBoost-delta", "PhyChronos-Irr"]):
        g = r[r.model == m].groupby("frac")["MAE"].mean()
        ax.plot(g.index * 100, g.values, lw=2, marker="o", ms=4, color=S[[1, 0][i]], label=m)
    zs = r[r.model.str.startswith("Chronos")]["MAE"].mean()
    ax.axhline(zs, color=S[2], lw=2, ls="--", label="Chronos-2+cov zero-shot")
    ax.set_xlabel("Share of training data used (%)")
    ax.set_ylabel("MAE 1-24 h (points)")
    ax.legend(fontsize=7.5)
    ax.set_title("Data efficiency", loc="left", fontsize=10, color=INK)
    save(fig, "fig_e10_data_efficiency")


ALL = dict(e1=fig_e1, e2=fig_e2, e3b=fig_e3b, e7=fig_e7, e6=fig_e6, e8=fig_e8, e10=fig_e10)
if __name__ == "__main__":
    for n in (sys.argv[1:] or ALL):
        try:
            ALL[n]()
        except Exception as e:
            print("skip", n, repr(e)[:150])
