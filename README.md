# Figure reproduction: magnetic and thermal properties of solar active regions

Scripts and data to reproduce Figures 1-7 of

Liu, J., Wang, Y., Ye, Y., Liu, L., Liu, R., and Wang, Y., Magnetic and Thermal Properties of Solar Active Regions in 14 Years.

Each `figN/` folder contains the arrays used in that figure (`figN_data.npz`) and a plotting script (`plot_figN.py`). The scripts only read the local npz file and draw the figure.

## Requirements

- Python 3.10 or later
- numpy, matplotlib, pandas, scipy

Figure 1 uses AIA colour tables from sunpy when it is installed (`pip install sunpy`). Without sunpy the maps are still drawn, with matplotlib fallback colormaps.

```bash
pip install -r requirements.txt
```

## Usage

From this directory:

```bash
python fig1/plot_fig1.py
python fig2/plot_fig2.py
python fig3/plot_fig3.py
python fig4/plot_fig4.py
python fig5/plot_fig5.py
python fig6/plot_fig6.py
python fig7/plot_fig7.py
```

Each script writes `figN.png` next to the script. Paths are relative to the script file, so the scripts can also be run from inside `figN/`.

## Layout

```
fig1/   Representative AR (SHARP 4375, 2014-07-20)
fig2/   Butterfly diagram (latitude vs time)
fig3/   PDFs of area, flux, and field strength
fig4/   Cycle evolution of area, flux, and AR count
fig5/   Three-band EM and Tw PDFs
fig6/   Three-band EM and Tw vs time
fig7/   Superhot-emission mask and Tw robustness
```

## Contact

Jiajia Liu (jiajialiu@ustc.edu.cn), University of Science and Technology of China.
