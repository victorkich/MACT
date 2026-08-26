"""Generate the project-page figures as pure-vector SVG (+ PNG companions).

Numbers are transcribed from Table 2 of the MACT paper (SMAC, median win rate over seeds,
50K env steps; 200K on 3s_vs_5z).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _f in ["Inter-Regular", "Inter-Medium", "Inter-SemiBold", "Inter-Bold"]:
    p = f"/usr/share/fonts/truetype/inter/{_f}.ttf"
    if os.path.exists(p):
        fm.fontManager.addfont(p)

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Inter",
    "svg.fonttype": "path",     # glyphs as outlines -> identical everywhere, still vector
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d1d5db",
    "axes.labelcolor": "#374151",
    "text.color": "#111111",
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

import os as _os
OUT = _os.environ.get("MACT_PLOTS_OUT",
                      _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                    "assets", "plots"))
os.makedirs(OUT, exist_ok=True)

METHODS = ["MACT", "MATWM", "MARIE", "MAMBA", "MBVD", "MAPPO"]
FAMILY = {"MACT": "Transformer WM", "MATWM": "Transformer WM", "MARIE": "Transformer WM",
          "MAMBA": "RNN WM", "MBVD": "RNN WM", "MAPPO": "Model-free"}
COL = {"MACT": "#1d4ed8", "MATWM": "#60a5fa", "MARIE": "#a5c4fc",
       "MAMBA": "#a78bfa", "MBVD": "#cdbcfb", "MAPPO": "#cbd5e1"}
FAMCOL = {"Transformer WM": "#60a5fa", "RNN WM": "#a78bfa", "Model-free": "#cbd5e1"}

MAPS = ["2m_vs_1z", "2s_vs_1sc", "2s3z", "3m", "3s_vs_3z",
        "3s_vs_4z", "8m", "MMM", "so_many_baneling", "3s_vs_5z"]
# map -> {method: (median, std)}
T2 = {
    "2m_vs_1z":         dict(MACT=(95.0, 0.0),  MATWM=(98.0, 3.2),  MARIE=(95.0, 4.4),  MAMBA=(91.0, 6.2),  MBVD=(41.0, 20.7), MAPPO=(51.0, 10.3)),
    "2s_vs_1sc":        dict(MACT=(98.7, 2.2),  MATWM=(96.0, 5.7),  MARIE=(90.0, 9.1),  MAMBA=(80.0, 7.3),  MBVD=(0.0, 1.2),   MAPPO=(18.0, 7.6)),
    "2s3z":             dict(MACT=(65.0, 1.7),  MATWM=(80.0, 9.0),  MARIE=(71.0, 8.6),  MAMBA=(68.0, 12.1), MBVD=(28.0, 17.5), MAPPO=(13.0, 3.0)),
    "3m":               dict(MACT=(96.7, 3.4),  MATWM=(83.0, 10.4), MARIE=(78.0, 14.1), MAMBA=(68.0, 7.7),  MBVD=(60.0, 9.2),  MAPPO=(54.0, 6.3)),
    "3s_vs_3z":         dict(MACT=(80.0, 3.4),  MATWM=(87.0, 19.4), MARIE=(85.0, 21.8), MAMBA=(77.0, 23.7), MBVD=(0.0, 0.0),   MAPPO=(0.0, 0.0)),
    "3s_vs_4z":         dict(MACT=(4.1, 7.2),   MATWM=(12.0, 4.8),  MARIE=(0.0, 0.8),   MAMBA=(4.0, 1.4),   MBVD=(0.0, 0.0),   MAPPO=(0.0, 0.0)),
    "8m":               dict(MACT=(95.0, 5.0),  MATWM=(67.0, 24.9), MARIE=(72.0, 7.1),  MAMBA=(68.0, 6.4),  MBVD=(52.0, 18.9), MAPPO=(38.0, 4.9)),
    "MMM":              dict(MACT=(39.2, 32.6), MATWM=(7.0, 4.7),   MARIE=(1.0, 1.6),   MAMBA=(3.0, 3.5),   MBVD=(0.0, 0.0),   MAPPO=(0.0, 0.0)),
    "so_many_baneling": dict(MACT=(94.4, 7.9),  MATWM=(86.0, 22.9), MARIE=(73.0, 12.4), MAMBA=(66.0, 14.2), MBVD=(27.0, 12.3), MAPPO=(31.0, 7.6)),
    "3s_vs_5z":         dict(MACT=(65.0, 11.7), MATWM=(64.0, 26.5), MARIE=(66.0, 28.0), MAMBA=(6.0, 10.1),  MBVD=(0.0, 0.0),   MAPPO=(0.0, 0.0)),
}
MEAN = dict(MACT=73.3, MATWM=68.0, MARIE=63.1, MAMBA=53.1, MBVD=20.8, MAPPO=20.5)
MEDIAN = dict(MACT=87.2, MATWM=81.5, MARIE=72.5, MAMBA=68.0, MBVD=13.5, MAPPO=15.5)


def save(fig, name):
    svg, png = f"{OUT}/{name}.svg", f"{OUT}/{name}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png, format="png", dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  {name}.svg  ({os.path.getsize(svg)//1024} KB)  + png")


def bar_labels(ax, bars, vals, fs=9.5):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.6, f"{v:.1f}",
                ha="center", va="bottom", fontsize=fs, color="#374151", fontweight="500")


def fig_headline():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5))
    for ax, (data, lab) in zip(axes, [(MEAN, "Mean win rate"), (MEDIAN, "Median win rate")]):
        vals = [data[m] for m in METHODS]
        cols = [COL[m] for m in METHODS]
        bars = ax.bar(range(len(METHODS)), vals, color=cols, width=0.68,
                      edgecolor=["#1e3a8a" if m == "MACT" else "none" for m in METHODS],
                      linewidth=[1.4 if m == "MACT" else 0 for m in METHODS], zorder=3)
        bar_labels(ax, bars, vals)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([f"{m}\n(ours)" if m == "MACT" else m for m in METHODS],
                           fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Win rate (%)", fontsize=10)
        ax.set_title(lab, fontsize=11.5, fontweight="600", pad=8)
        ax.grid(axis="y", color="#eef0f3", lw=0.9, zorder=0)
        ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMCOL.values()]
    fig.legend(handles, list(FAMCOL.keys()), loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    save(fig, "fig_headline")


def fig_permap():
    fig, ax = plt.subplots(figsize=(12.4, 4.0))
    n = len(METHODS)
    w = 0.8 / n
    xs = np.arange(len(MAPS))
    for j, m in enumerate(METHODS):
        vals = [T2[k][m][0] for k in MAPS]
        errs = [T2[k][m][1] for k in MAPS]
        off = (j - (n - 1) / 2) * w
        ax.bar(xs + off, vals, width=w * 0.92, color=COL[m], label=m, zorder=3,
               edgecolor="#1e3a8a" if m == "MACT" else "none",
               linewidth=1.1 if m == "MACT" else 0)
        ax.errorbar(xs + off, vals, yerr=errs, fmt="none", ecolor="#9ca3af",
                    elinewidth=0.7, capsize=1.6, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(MAPS, fontsize=9.5,
                       rotation=18, ha="right")
    ax.set_ylabel("Median win rate (%)", fontsize=10)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color="#eef0f3", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, 1.14))
    ax.text(0.995, -0.30, "3s_vs_5z at 200K steps; all others 50K",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#9ca3af")
    fig.tight_layout()
    save(fig, "fig_permap")


def fig_margin():
    """MACT minus the best baseline, per map -- where coordination pressure is highest."""
    deltas = []
    for k in MAPS:
        best = max(T2[k][m][0] for m in METHODS if m != "MACT")
        deltas.append(T2[k]["MACT"][0] - best)
    order = np.argsort(deltas)
    maps_s = [MAPS[i] for i in order]
    d_s = [deltas[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    cols = ["#1d4ed8" if d > 0 else "#e5a3a3" for d in d_s]
    bars = ax.barh(range(len(maps_s)), d_s, color=cols, height=0.66, zorder=3)
    for b, d in zip(bars, d_s):
        ax.text(d + (1.1 if d >= 0 else -1.1), b.get_y() + b.get_height() / 2,
                f"{d:+.1f}", va="center", ha="left" if d >= 0 else "right",
                fontsize=9.5, color="#374151", fontweight="500")
    ax.axvline(0, color="#9ca3af", lw=1.0, zorder=4)
    ax.set_yticks(range(len(maps_s)))
    ax.set_yticklabels(maps_s, fontsize=9.5)
    ax.set_xlabel("MACT − best baseline  (win-rate points)", fontsize=10)
    ax.set_xlim(min(d_s) - 9, max(d_s) + 9)
    ax.grid(axis="x", color="#eef0f3", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_margin")


if __name__ == "__main__":
    print("figures ->", OUT)
    fig_headline()
    fig_permap()
    fig_margin()


# ── Reproduction check ─────────────────────────────────────────────────────────
# Win rates measured here by re-running the released checkpoints under the paper's
# protocol (30 evaluation episodes per seed, median across seeds), against Table 2.
REPRO = {
    # map: {method: (ours_median, ours_std, n_seeds)}
    "so_many_baneling": {"MACT": (100.0, 1.9, 3), "MARIE": (36.7, 0.0, 1)},
    "3s_vs_3z":         {"MACT": (86.7, 10.7, 3), "MARIE": (96.7, 0.0, 1)},
    "MMM":              {"MACT": (23.3, 33.0, 2), "MARIE": (6.7, 0.0, 1)},
    "8m":               {"MACT": (46.7, 0.0, 1),  "MARIE": (46.7, 0.0, 1)},
}
REPRO_MAPS = ["so_many_baneling", "3s_vs_3z", "MMM", "8m"]


def fig_repro():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), sharey=True)
    for ax, method in zip(axes, ["MACT", "MARIE"]):
        xs = np.arange(len(REPRO_MAPS))
        pub = [T2[m][method][0] for m in REPRO_MAPS]
        pub_e = [T2[m][method][1] for m in REPRO_MAPS]
        our = [REPRO[m][method][0] for m in REPRO_MAPS]
        our_e = [REPRO[m][method][1] for m in REPRO_MAPS]
        ns = [REPRO[m][method][2] for m in REPRO_MAPS]

        ax.bar(xs - 0.19, pub, width=0.36, color="#cbd5e1", label="paper (Table 2)", zorder=3)
        ax.errorbar(xs - 0.19, pub, yerr=pub_e, fmt="none", ecolor="#94a3b8",
                    elinewidth=0.9, capsize=2.2, zorder=4)
        col = "#1d4ed8" if method == "MACT" else "#a5c4fc"
        ax.bar(xs + 0.19, our, width=0.36, color=col, label="reproduced here", zorder=3)
        ax.errorbar(xs + 0.19, our, yerr=our_e, fmt="none", ecolor="#64748b",
                    elinewidth=0.9, capsize=2.2, zorder=4)
        for x, v, n in zip(xs + 0.19, our, ns):
            ax.text(x, v + 3.0, f"n={n}", ha="center", va="bottom",
                    fontsize=8, color="#6b7280")

        ax.set_xticks(xs)
        ax.set_xticklabels(REPRO_MAPS, fontsize=9, rotation=14, ha="right")
        ax.set_ylim(0, 116)
        ax.set_title(f"{method}{'  (ours)' if method == 'MACT' else ''}",
                     fontsize=11.5, fontweight="600", pad=8)
        ax.grid(axis="y", color="#eef0f3", lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].set_ylabel("Win rate (%)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_repro")


if __name__ == "__main__":
    fig_repro()
