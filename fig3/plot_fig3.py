# Read the data and plot Figure 3.
# Jiajia Liu

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "fig3_data.npz")


def _fmt(v, kind):
    if not np.isfinite(v):
        return "NA"
    if kind == "log":
        val = 10.0 ** v
        mant, exp = f"{val:.2e}".split("e")
        return rf"${float(mant):.2f}\times 10^{{{int(exp)}}}$"
    if kind == "sci":
        mant, exp = f"{v:.2e}".split("e")
        return rf"${float(mant):.2f}\times 10^{{{int(exp)}}}$"
    if kind == "count":
        return f"{v:.1f}"
    return f"{v:.1f}"


panels = [
    (np.asarray(d["area"], dtype=float), "AR area", r"$\log_{{10}}(\mathrm{\mu Hem})$", "log", True, "microHem"),
    (np.asarray(d["usflux"], dtype=float), "Unsigned flux (USFlux)", r"$\log_{{10}}(\mathrm{Mx})$", "log", True, "Mx"),
    (np.asarray(d["netflux"], dtype=float), "Signed net flux (NetFlux)", "Mx", "sci", False, "Mx"),
    (np.asarray(d["bz_mean"], dtype=float), r"Signed mean $B_z$ ($|B_z|\geq150$ G)", "Gauss", "G", False, "G"),
    (np.asarray(d["abs_bz_mean"], dtype=float), r"Mean $|B_z|$ ($|B_z|\geq150$ G)", "Gauss", "G", False, "G"),
    (np.asarray(d["monthly_ar_count"], dtype=float), "Monthly AR count", "number of ARs", "count", False, ""),
]
colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#B07AA1"]

plt.style.use("seaborn-v0_8-white")
fig, axes = plt.subplots(3, 2, figsize=(14.4, 12.0), constrained_layout=True)
axes = np.array(axes).reshape(-1)

for i, (ax, (x, title, xlabel, kind, is_log, unit)) in enumerate(zip(axes, panels)):
    raw = x[np.isfinite(x)]
    if is_log:
        raw = raw[raw > 0]
        x = np.log10(raw)
        x_mean = float(np.log10(np.mean(raw)))
        x_median = float(np.log10(np.median(raw)))
        bins = 40
    elif kind == "count":
        x = raw
        x_mean = float(np.mean(raw))
        x_median = float(np.median(raw))
        bins = max(12, min(24, len(np.unique(x))))
    else:
        p = np.nanpercentile(np.abs(raw), 99.5)
        x = raw.clip(-p, p)
        x_mean = float(np.mean(raw))
        x_median = float(np.median(raw))
        bins = 45
    color = colors[i]
    ax.hist(x, bins=bins, density=True, color=color, alpha=0.55, edgecolor="#f5f5f5", linewidth=0.5)
    kde = gaussian_kde(x)
    xline = np.linspace(np.nanmin(x), np.nanmax(x), 240)
    yline = kde(xline)
    ax.plot(xline, yline, color=color, linewidth=2.0)
    x_peak = float(xline[int(np.nanargmax(yline))])
    ax.axvline(x_peak, color="black", linestyle="-", linewidth=1.2)
    ax.axvline(x_mean, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(x_median, color="black", linestyle="-.", linewidth=1.2)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    suffix = f" {unit}" if unit else ""
    ax.text(
        x0 + 0.02 * (x1 - x0),
        y0 + 0.98 * (y1 - y0),
        f"peak: {_fmt(x_peak, kind)}{suffix}\nmean: {_fmt(x_mean, kind)}{suffix}\nmedian: {_fmt(x_median, kind)}{suffix}",
        color="black",
        fontsize=10.5,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.6},
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("PDF", fontsize=11)
    ax.set_title(f"{chr(ord('a') + i)}) {title}", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

out = HERE / "fig3.png"
fig.savefig(out, dpi=260)
plt.close(fig)
print(out)
