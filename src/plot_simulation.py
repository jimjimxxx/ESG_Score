"""畫模擬結果：四個情境的等級相關，以及雜訊指標比例掃描。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulate import METHODS

ROOT = Path(__file__).resolve().parent.parent
# 三個實際方法用類別色前三格；Oracle 只是天花板參考 → 中性灰、空心
COLOR = {METHODS[0]: "#2a78d6", METHODS[1]: "#eb6834", METHODS[2]: "#1baf7a", METHODS[3]: "#8a8984"}
MARKER = {METHODS[0]: "o", METHODS[1]: "s", METHODS[2]: "D", METHODS[3]: "o"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.sans-serif": ["Calibri", "DejaVu Sans"], "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10,
})


def style(m):
    hollow = m == METHODS[3]
    return dict(color=COLOR[m], marker=MARKER[m], markersize=8,
                markerfacecolor="white" if hollow else COLOR[m], markeredgewidth=1.6)


def main():
    sc = pd.read_csv(ROOT / "results" / "simulation_raw.csv")
    sw = pd.read_csv(ROOT / "results" / "simulation_noise_sweep.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))

    ax = axes[0]
    scen = list(dict.fromkeys(sc["scenario"]))
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    for k, m in enumerate(METHODS):
        g = sc[sc["method"] == m].groupby("scenario")["spearman"].agg(["mean", "std"]).reindex(scen)
        ax.errorbar(np.arange(len(scen)) + offsets[k], g["mean"], yerr=g["std"], ls="none",
                    ecolor=COLOR[m], elinewidth=1.2, capsize=0, label=m, **style(m))
    ax.set_xticks(range(len(scen)), [s.split(" ", 1)[1].replace(", ", ",\n") for s in scen])
    ax.set_ylabel("Spearman correlation with true ESG quality")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Recovery of true ESG quality (20 replications, mean ± sd)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower left", fontsize=9)

    ax = axes[1]
    for m in METHODS:
        g = sw[sw["method"] == m].groupby("noise_frac")["spearman"].agg(["mean", "std"])
        ax.errorbar(g.index * 100, g["mean"], yerr=g["std"], lw=2, elinewidth=1.2, capsize=0,
                    label=m, **style(m))
    ax.set_xlabel("Share of pure-noise indicators (%)")
    ax.set_ylabel("Spearman correlation with true ESG quality")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Sweep: adding noise indicators (10 replications)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "simulation.png", dpi=180)
    print("saved figures/simulation.png")


if __name__ == "__main__":
    main()
