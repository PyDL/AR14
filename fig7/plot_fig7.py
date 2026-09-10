# Read the data and plot Figure 7.
# Jiajia Liu

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "fig7_data.npz")
shown = np.asarray(d["em_log10"], dtype=float)
perm = np.asarray(d["permissive"], dtype=bool)
cons = np.asarray(d["conservative"], dtype=bool)
tw_t = pd.to_datetime(np.asarray(d["tw_time"]))
sun = pd.Series(np.asarray(d["sunspot"], dtype=float), index=pd.to_datetime(np.asarray(d["sunspot_time"]))).sort_index()
series = [
    (np.asarray(d["tw_all"], dtype=float), "all, unmasked", "#1f77b4"),
    (np.asarray(d["tw_nonempty"], dtype=float), "nonempty, unmasked", "#2ca02c"),
    (np.asarray(d["tw_permissive"], dtype=float), "nonempty, permissive masked", "#ff7f0e"),
    (np.asarray(d["tw_conservative"], dtype=float), "nonempty, conservative masked", "black"),
]


def _cc(y):
    s = pd.Series(y, index=tw_t)
    joined = pd.concat([s.rename("a"), sun.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 12:
        return np.nan
    return float(spearmanr(joined["a"], joined["b"]).statistic)


plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8})
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), gridspec_kw={"width_ratios": [1.0, 1.0]})
fig.subplots_adjust(left=0.055, right=0.93, bottom=0.15, top=0.88, wspace=0.30)
finite = shown[np.isfinite(shown)]
low, high = np.nanpercentile(finite, [5, 99.5])
axes[0].imshow(shown, origin="lower", cmap="magma", vmin=low, vmax=high, aspect="auto")
if np.any(perm):
    axes[0].contour(perm.astype(float), levels=[0.5], colors="yellow", linewidths=0.7)
if np.any(cons):
    axes[0].contour(cons.astype(float), levels=[0.5], colors="black", linewidths=0.95)
axes[0].set_title("(a) HARP 4379, 8–12 MK EM")
axes[0].set_xticks([])
axes[0].set_yticks([])

ax = axes[1]
for y, name, color in series:
    ax.plot(tw_t, y, color=color, lw=1.8, label=f"{name} (CC={_cc(y):.2f})")
ax2 = ax.twinx()
ax2.plot(sun.index, sun.to_numpy(), color="#d62728", ls="--", lw=1.1, alpha=0.85, label="sunspot number")
ax2.set_ylabel("Sunspot number")
ax.set_ylabel(r"8–12 MK $T_w$ (MK)")
ax.set_title(r"(b) Sample-controlled versus spatially masked $T_w$")
handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles + handles2, labels + labels2, frameon=False, fontsize=7, loc="upper left")
ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.canvas.draw()
pos_right = axes[1].get_position()
pos_left = axes[0].get_position()
axes[0].set_position([pos_left.x0, pos_right.y0, pos_right.width, pos_right.height])
ax2.set_position(axes[1].get_position())

out = HERE / "fig7.png"
fig.savefig(out, dpi=200)
plt.close(fig)
print(out)
