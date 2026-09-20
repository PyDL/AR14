# Read the data and plot Figure 5.
# Jiajia Liu

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "fig5_data.npz")


def _clip(x, lo=0.5, hi=99.5):
    a, b = np.nanpercentile(x, [lo, hi])
    return x[(x >= a) & (x <= b)]


def _fmt(v, is_log, unit):
    if not np.isfinite(v):
        return "NA"
    if is_log:
        val = 10.0 ** v
        mant, exp = f"{val:.2e}".split("e")
        return rf"${float(mant):.2f}\times 10^{{{int(exp)}}}$ {unit}"
    return f"{v:.2f} {unit}"


em_panels = [
    (np.asarray(d["em_warm"], dtype=float), "EM Total (T < 3 MK)"),
    (np.asarray(d["em_hot"], dtype=float), "EM Total (3-8 MK)"),
    (np.asarray(d["em_superhot"], dtype=float), "EM Total (8-12 MK)"),
]
tw_panels = [
    (np.asarray(d["tw_warm"], dtype=float) / 1e6, "Tw (T < 3 MK)"),
    (np.asarray(d["tw_hot"], dtype=float) / 1e6, "Tw (3-8 MK)"),
    (np.asarray(d["tw_superhot"], dtype=float) / 1e6, "Tw (8-12 MK)"),
]
colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#B07AA1"]

plt.style.use("seaborn-v0_8-white")
fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.1), constrained_layout=True)

for i, (x, title) in enumerate(em_panels):
    ax = axes[0, i]
    color = colors[i]
    raw = x[np.isfinite(x) & (x > 0)]
    x = _clip(np.log10(raw))
    ax.hist(x, bins=40, density=True, color=color, alpha=0.55, edgecolor="#f5f5f5", linewidth=0.5)
    kde = gaussian_kde(x)
    xline = np.linspace(np.nanmin(x), np.nanmax(x), 240)
    yline = kde(xline)
    ax.plot(xline, yline, color=color, linewidth=2.0)
    x_peak = float(xline[int(np.nanargmax(yline))])
    x_mean = float(np.log10(np.mean(raw)))
    x_median = float(np.log10(np.median(raw)))
    ax.axvline(x_peak, color="black", linestyle="-", linewidth=1.2)
    ax.axvline(x_mean, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(x_median, color="black", linestyle="-.", linewidth=1.2)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.text(
        x0 + 0.02 * (x1 - x0),
        y0 + 0.98 * (y1 - y0),
        f"peak: {_fmt(x_peak, True, r'cm$^{-3}$')}\nmean: {_fmt(x_mean, True, r'cm$^{-3}$')}\nmedian: {_fmt(x_median, True, r'cm$^{-3}$')}",
        fontsize=9.8,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.6},
    )
    ax.set_title(f"{chr(ord('a') + i)}) {title}", fontsize=13)
    ax.set_xlabel(r"$\log_{10}(\mathrm{cm}^{-3})$", fontsize=11)
    ax.set_ylabel("PDF", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for i, (x, title) in enumerate(tw_panels):
    ax = axes[1, i]
    color = colors[3 + i]
    raw = x[np.isfinite(x)]
    x = _clip(raw)
    ax.hist(x, bins=40, density=True, color=color, alpha=0.55, edgecolor="#f5f5f5", linewidth=0.5)
    kde = gaussian_kde(x)
    xline = np.linspace(np.nanmin(x), np.nanmax(x), 240)
    yline = kde(xline)
    ax.plot(xline, yline, color=color, linewidth=2.0)
    x_peak = float(xline[int(np.nanargmax(yline))])
    x_mean = float(np.mean(raw))
    x_median = float(np.median(raw))
    ax.axvline(x_peak, color="black", linestyle="-", linewidth=1.2)
    ax.axvline(x_mean, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(x_median, color="black", linestyle="-.", linewidth=1.2)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ha = "right" if i == 1 else "left"
    xpos = x0 + (0.98 if i == 1 else 0.02) * (x1 - x0)
    ax.text(
        xpos,
        y0 + 0.98 * (y1 - y0),
        f"peak: {_fmt(x_peak, False, 'MK')}\nmean: {_fmt(x_mean, False, 'MK')}\nmedian: {_fmt(x_median, False, 'MK')}",
        fontsize=9.8,
        ha=ha,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.6},
    )
    ax.set_title(f"{chr(ord('d') + i)}) {title}", fontsize=13)
    ax.set_xlabel("MK", fontsize=11)
    ax.set_ylabel("PDF", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

out = HERE / "fig5.png"
fig.savefig(out, dpi=260)
plt.close(fig)
print(out)
