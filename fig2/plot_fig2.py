# Read the data and plot Figure 2.
# Jiajia Liu

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "fig2_data.npz")
time = pd.to_datetime(np.asarray(d["time"]))
lat = np.asarray(d["lat"], dtype=float)
area = np.asarray(d["area"], dtype=float)
usflux = np.asarray(d["usflux"], dtype=float)
em_total = np.asarray(d["em_total"], dtype=float)

valid = np.isfinite(area) & (area > 0)
lo, hi = np.nanpercentile(area[valid], [5, 95])
sizes = 12.0 + (np.clip(area, lo, hi) - lo) / (hi - lo) * (220.0 - 12.0)
sizes[~valid] = 12.0


def _bounds(x):
    vmin = np.nanpercentile(x, 2)
    vmax = np.nanpercentile(x, 98)
    if not np.isfinite(vmin) or vmin <= 0:
        vmin = np.nanmin(x[x > 0])
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = np.nanmax(x)
    return float(vmin), float(vmax)


umin, umax = _bounds(usflux)
emin, emax = _bounds(em_total)

fig, axes = plt.subplots(2, 1, figsize=(13.6, 9.8), sharex=True, sharey=True)
sc0 = axes[0].scatter(
    time, lat, s=sizes, c=usflux, cmap="plasma",
    norm=LogNorm(vmin=umin, vmax=umax), alpha=0.75, edgecolors="black", linewidths=0.15,
)
cb0 = fig.colorbar(sc0, ax=axes[0], pad=0.02, fraction=0.035)
cb0.set_label("USFLUX [Mx]")
sc1 = axes[1].scatter(
    time, lat, s=sizes, c=em_total, cmap="cividis",
    norm=LogNorm(vmin=emin, vmax=emax), alpha=0.75, edgecolors="black", linewidths=0.15,
)
cb1 = fig.colorbar(sc1, ax=axes[1], pad=0.02, fraction=0.035)
cb1.set_label(r"Total EM [cm$^{-3}$]")
for ax in axes:
    ax.axhline(0, color="gray", linewidth=0.9, alpha=0.7)
    ax.set_ylim(-45, 45)
    ax.set_ylabel("Latitude (deg)")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=35)
axes[1].set_xlabel("Time")
axes[0].set_title("a) USFLUX")
axes[1].set_title("b) Total EM")
q25, q50, q75 = np.nanpercentile(area[valid], [25, 50, 75])
demo = 12.0 + (np.clip(np.array([q25, q50, q75]), lo, hi) - lo) / (hi - lo) * (220.0 - 12.0)
handles = [
    axes[0].scatter([], [], s=s, c="gray", alpha=0.5, edgecolors="black", linewidths=0.15)
    for s in demo
]
axes[0].legend(
    handles,
    [f"area P25={q25:.1f} microHem", f"area P50={q50:.1f} microHem", f"area P75={q75:.1f} microHem"],
    title="Marker size (AR area)",
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    frameon=True,
    fontsize=8,
    title_fontsize=9,
)
fig.subplots_adjust(left=0.08, right=0.94, bottom=0.10, top=0.97, hspace=0.16)
out = HERE / "fig2.png"
fig.savefig(out, dpi=260)
plt.close(fig)
print(out)
