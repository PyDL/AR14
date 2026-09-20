# Read the data and plot Figure 6.
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
d = np.load(HERE / "fig6_data.npz")
time = pd.to_datetime(np.asarray(d["time"]))
sun = pd.Series(np.asarray(d["sunspot"], dtype=float), index=pd.to_datetime(np.asarray(d["sunspot_time"]))).sort_index()
panels = [
    (np.asarray(d["em_warm"], dtype=float), r"EM total ($T<3$ MK)"),
    (np.asarray(d["em_hot"], dtype=float), "EM total (3–8 MK)"),
    (np.asarray(d["em_superhot"], dtype=float), "EM total (8–12 MK)"),
    (np.asarray(d["emavg_warm"], dtype=float), r"Average EM ($T<3$ MK)"),
    (np.asarray(d["emavg_hot"], dtype=float), "Average EM (3–8 MK)"),
    (np.asarray(d["emavg_superhot"], dtype=float), "Average EM (8–12 MK)"),
    (np.asarray(d["tw_warm"], dtype=float), r"$T_w$ ($T<3$ MK)"),
    (np.asarray(d["tw_hot"], dtype=float), r"$T_w$ (3–8 MK)"),
    (np.asarray(d["tw_superhot"], dtype=float), r"$T_w$ (8–12 MK)"),
]


def _cc(a, b):
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 12:
        return np.nan
    return float(spearmanr(joined["a"], joined["b"]).statistic)


plt.style.use("seaborn-v0_8-white")
fig, axes = plt.subplots(3, 3, figsize=(16.0, 9.0), sharex=True, constrained_layout=True)
axes = np.array(axes).reshape(-1)

for i, (ax, (x, title)) in enumerate(zip(axes, panels)):
    m = np.isfinite(x)
    ax.scatter(time[m], x[m], s=9, color="#1f77b4", alpha=0.35, edgecolors="none", zorder=1)
    trend_df = pd.DataFrame({"time": time[m], "x": x[m]}).dropna()
    trend_df["month"] = trend_df["time"].dt.to_period("M").dt.to_timestamp()
    trend = (
        trend_df.groupby("month")["x"]
        .mean()
        .sort_index()
        .reindex(sun.index)
    )
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
    cc_p = _cc(upper, sun)
    l1 = ax.plot(trend_6m.index, trend_6m.to_numpy(), color="#1f77b4", linewidth=2.2, zorder=3)
    l2 = ax.plot(upper.index, upper.to_numpy(), color="#CC79A7", linewidth=1.7, alpha=0.95, zorder=3)
    ax_r = ax.twinx()
    l3 = ax_r.plot(sun.index, sun.to_numpy(), color="#d62728", linewidth=1.8, linestyle="--")
    ax_r.set_ylabel("Sunspot number", color="#d62728", fontsize=9.0)
    ax_r.tick_params(axis="y", labelcolor="#d62728", labelsize=8.4)
    ax_r.spines["top"].set_visible(False)
    ax.set_title(f"{chr(ord('a') + i)}) {title}", fontsize=11.8)
    ax.set_ylabel(title, color="#1f77b4", fontsize=9.0)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=8.4)
    ax.legend(
        list(l1) + list(l2) + list(l3),
        [f"6-month-smoothed monthly mean (CC={cc_b:.2f})", f"Upper envelope (CC={cc_p:.2f})", "Sunspot number"],
        loc="upper left",
        fontsize=8.2,
        frameon=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=30, labelsize=9.0, labelbottom=True)
for ax in axes[-3:]:
    ax.set_xlabel("Time", fontsize=10.0)

out = HERE / "fig6.png"
fig.savefig(out, dpi=260)
plt.close(fig)
print(out)
