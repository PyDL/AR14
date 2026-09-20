# Read the data and plot Figure 4.
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
d = np.load(HERE / "fig4_data.npz")
time = pd.to_datetime(np.asarray(d["time"]))
sun_t = pd.to_datetime(np.asarray(d["sunspot_time"]))
sun = pd.Series(np.asarray(d["sunspot"], dtype=float), index=sun_t).sort_index()
values = {
    "AR Area [microHem]": np.asarray(d["area"], dtype=float),
    "USFlux [Mx]": np.asarray(d["usflux"], dtype=float),
    "NetFlux [Mx]": np.asarray(d["netflux"], dtype=float),
    "Active Region Count [count]": np.asarray(d["ar_count"], dtype=float),
}


def _cc(a, b):
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 12:
        return np.nan
    return float(spearmanr(joined["a"], joined["b"]).statistic)


plt.style.use("seaborn-v0_8-white")
fig, axes = plt.subplots(2, 2, figsize=(16.0, 9.0), sharex=True, constrained_layout=True)
axes = np.array(axes).reshape(-1)

for i, (ax, (title, x)) in enumerate(zip(axes, values.items())):
    m = np.isfinite(x)
    ax.scatter(time[m], x[m], s=10, color="#1f77b4", alpha=0.35, edgecolors="none", zorder=1)
    trend_df = pd.DataFrame({"time": time[m], "x": x[m]}).dropna()
    trend_df["month"] = trend_df["time"].dt.to_period("M").dt.to_timestamp()
    trend = (
        trend_df.groupby("month")["x"]
        .mean()
        .sort_index()
        .reindex(sun.index)
    )
    if i == 3:
        trend = trend.fillna(0.0)
    upper = (
        trend_df.groupby("month")["x"]
        .quantile(0.95)
        .sort_index()
        .reindex(sun.index)
        .rolling(window=6, center=True, min_periods=4)
        .median()
    )
    trend_6m = trend.rolling(window=6, center=True, min_periods=4).mean()
    cc_b = _cc(trend_6m, sun)
    legend_plain = title.split(" [", 1)[0]
    l1 = ax.plot(trend_6m.index, trend_6m.to_numpy(), color="#1f77b4", linewidth=2.2, zorder=3)
    handles = list(l1)
    labels = [f"{legend_plain} 6-month-smoothed monthly mean (CC={cc_b:.2f})"]
    if i != 3:
        cc_p = _cc(upper, sun)
        l2 = ax.plot(upper.index, upper.to_numpy(), color="#CC79A7", linewidth=1.7, alpha=0.95, zorder=3)
        handles += list(l2)
        labels.append(f"{legend_plain} upper envelope (CC={cc_p:.2f})")
        y_scatter = x[m]
        y_hi = max(float(np.nanpercentile(y_scatter, 99.5)), float(np.nanmax(upper)), float(np.nanmax(trend_6m)))
        if i == 2:
            span = max(float(np.nanpercentile(np.abs(y_scatter), 99.5)), float(np.nanmax(np.abs(upper))), float(np.nanmax(np.abs(trend_6m))))
            ax.set_ylim(-span * 1.08, span * 1.08)
        else:
            ax.set_ylim(0.0, y_hi * 1.08)
    ax.set_ylabel(title, color="#1f77b4", fontsize=9.2)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=8.8)
    ax.set_title(f"{chr(ord('a') + i)}) {legend_plain}", fontsize=12)
    ax_r = ax.twinx()
    l3 = ax_r.plot(sun.index, sun.to_numpy(), color="#d62728", linewidth=1.8, linestyle="--")
    ax_r.set_ylabel("Sunspot number", color="#d62728", fontsize=9.2)
    ax_r.tick_params(axis="y", labelcolor="#d62728", labelsize=8.8)
    ax_r.spines["top"].set_visible(False)
    handles += list(l3)
    labels.append("Sunspot number")
    ax.legend(handles, labels, loc="upper left", fontsize=8.5, frameon=True, edgecolor="#dddddd")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=30, labelsize=9.2, labelbottom=True)
for ax in axes[2:]:
    ax.set_xlabel("Time", fontsize=10.5)

out = HERE / "fig4.png"
fig.savefig(out, dpi=260)
plt.close(fig)
print(out)
