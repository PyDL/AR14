# Read the data and plot Figure 1.
# Jiajia Liu

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from sunpy.visualization.colormaps import cm as _sunpy_cm
except Exception:
    _sunpy_cm = None

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "fig1_data.npz", allow_pickle=True)
extent = tuple(d["extent"].tolist())
bz = np.asarray(d["bz"], dtype=float)
bt = np.asarray(d["bt"], dtype=float)
harp = int(np.asarray(d["harp"]).reshape(-1)[0])
noaa = str(np.asarray(d["noaa"]).reshape(-1)[0])
chosen_time = str(np.asarray(d["chosen_time"]).reshape(-1)[0])

aia_cmaps = {
    "304": "sdoaia304" if _sunpy_cm is not None else "Reds",
    "171": "sdoaia171" if _sunpy_cm is not None else "YlOrBr",
    "131": "sdoaia131" if _sunpy_cm is not None else "GnBu",
    "211": "sdoaia211" if _sunpy_cm is not None else "Purples",
    "335": "sdoaia335" if _sunpy_cm is not None else "Blues",
    "193": "sdoaia193" if _sunpy_cm is not None else "Oranges",
    "94": "sdoaia94" if _sunpy_cm is not None else "hot",
}

panels = [
    ("Bz [G]", bz, "RdBu_r", "Bz (G)", (-500.0, 500.0), False),
    ("Bt [G]", bt, "cividis", "Bt (G)", (100.0, 800.0), False),
    ("AIA 304", np.asarray(d["aia_304"], dtype=float), aia_cmaps["304"], "Intensity (DN)", (1.0, 50.0), True),
    ("AIA 171", np.asarray(d["aia_171"], dtype=float), aia_cmaps["171"], "Intensity (DN)", (10.0, 1000.0), True),
    ("AIA 131", np.asarray(d["aia_131"], dtype=float), aia_cmaps["131"], "Intensity (DN)", (1.0, 50.0), True),
    ("AIA 211", np.asarray(d["aia_211"], dtype=float), aia_cmaps["211"], "Intensity (DN)", (5.0, 700.0), True),
    ("AIA 335", np.asarray(d["aia_335"], dtype=float), aia_cmaps["335"], "Intensity (DN)", (1.0, 50.0), True),
    ("AIA 193", np.asarray(d["aia_193"], dtype=float), aia_cmaps["193"], "Intensity (DN)", (100.0, 2000.0), True),
    ("AIA 94", np.asarray(d["aia_94"], dtype=float), aia_cmaps["94"], "Intensity (DN)", (1.0, 30.0), True),
    ("EM (T < 3 MK)", np.asarray(d["em_warm"], dtype=float), "OrRd", r"EM (cm$^{-5}$)", None, False),
    ("EM (3-8 MK)", np.asarray(d["em_hot"], dtype=float), "OrRd", r"EM (cm$^{-5}$)", None, False),
    ("EM (8-12 MK)", np.asarray(d["em_superhot"], dtype=float), "OrRd", r"EM (cm$^{-5}$)", None, False),
]

fig, axes = plt.subplots(6, 2, figsize=(13.4, 20.4))
fig.subplots_adjust(left=0.07, right=0.94, top=0.968, bottom=0.019, wspace=0.30, hspace=0.009)
axs = np.array(axes).reshape(-1)
ny, nx = bz.shape
xx, yy = np.meshgrid(
    np.linspace(extent[0], extent[1], nx),
    np.linspace(extent[2], extent[3], ny),
)

for i, (title, arr, cmap, cbar_lab, frange, is_log) in enumerate(panels):
    ax = axs[i]
    kwargs = {"origin": "lower", "cmap": cmap, "extent": extent}
    finite = arr[np.isfinite(arr)]
    if frange is not None:
        vmin, vmax = frange
        if is_log:
            kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
        else:
            kwargs["vmin"], kwargs["vmax"] = vmin, vmax
    else:
        pos = finite[finite > 0]
        vmin = np.nanpercentile(pos, 5)
        vmax = np.nanpercentile(pos, 99.5)
        kwargs["norm"] = LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin * 1.0001))
    im = ax.imshow(arr, **kwargs)
    if np.nanmax(bz) >= 150:
        ax.contour(xx, yy, bz, levels=[150], colors=["purple"], linewidths=2.0)
    if np.nanmin(bz) <= -150:
        ax.contour(xx, yy, bz, levels=[-150], colors=["limegreen"], linewidths=2.0)
    ax.set_title(f"{chr(ord('a') + i)}) {title}", fontsize=14.0)
    ax.set_xlabel("X [Mm]", fontsize=10.8)
    ax.set_ylabel("Y [Mm]", fontsize=10.8)
    ax.tick_params(labelsize=10.0)
    cax = make_axes_locatable(ax).append_axes("right", size="4.2%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=9.6)
    cb.set_label(cbar_lab, fontsize=9.8)

fig.suptitle(f"{chosen_time[:8]}  SHARP {harp}  (NOAA {noaa})", fontsize=18)
out = HERE / "fig1.png"
fig.savefig(out, dpi=210)
plt.close(fig)
print(out)
