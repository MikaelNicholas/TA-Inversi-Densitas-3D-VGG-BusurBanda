"""
=============================================================
  Visualisasi Hasil GRABlox2 — Banda Arc
  FULL VERSION + shapefile lokal + kedalaman seragam
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter1d
import xarray as xr
import warnings
import glob
import re
import json
import urllib.request
import os
import shapefile   # <-- untuk baca shapefile
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────
# KONFIGURASI GLOBAL
# ───────────────────────────────────────────────
FILE_DAT = "Grablox2_current_iter.dat"
FILE_GAT = "Grablox2_current_iter.gat"
FILE_BLX = "Grablox2_current_iter.blx"

NX_DATA = 97
NY_DATA = 55
X_DATA_MIN, X_DATA_MAX = 4.554, 1784.715
Y_DATA_MIN, Y_DATA_MAX = 4.638, 1006.518

LON_MIN, LON_MAX = 118.0, 134.0
LAT_MIN, LAT_MAX = -11.0, -2.0

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'figure.dpi': 150,
})

# ───────────────────────────────────────────────
# BACA DATA
# ───────────────────────────────────────────────
print("Membaca file .dat ...")
dat_rows = []
with open(FILE_DAT) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            float(line.split()[0])
            dat_rows.append(line)
        except:
            continue

df_dat = pd.DataFrame(
    [r.split() for r in dat_rows if len(r.split()) >= 6],
    columns=['x', 'y', 'h', 'g_comp', 'g_base', 'g_meas']
).apply(pd.to_numeric, errors='coerce').dropna()

df_dat = df_dat[
    (df_dat.x >= X_DATA_MIN - 1) & (df_dat.x <= X_DATA_MAX + 1) &
    (df_dat.y >= Y_DATA_MIN - 1) & (df_dat.y <= Y_DATA_MAX + 1)
]
df_dat['g_meas_corr'] = df_dat['g_meas'] - df_dat['g_base']
df_dat['residual'] = df_dat['g_meas_corr'] - df_dat['g_comp']
print(f"  DAT: {len(df_dat)} titik")

print("Membaca file .gat ...")
gat_rows = []
with open(FILE_GAT) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            vals = [float(v) for v in line.split()]
            if len(vals) >= 5:
                gat_rows.append(vals)
        except:
            continue

df_gat = pd.DataFrame(gat_rows)
if df_gat.shape[1] >= 5:
    df_gat = df_gat.iloc[:, 0:5]
    df_gat.columns = ['x', 'y', 'h', 'gzz_comp', 'gzz_meas']
else:
    df_gat.columns = ['x', 'y', 'h', 'gzz_comp']
    df_gat['gzz_meas'] = np.nan

df_gat = df_gat[
    (df_gat.x >= X_DATA_MIN - 1) & (df_gat.x <= X_DATA_MAX + 1) &
    (df_gat.y >= Y_DATA_MIN - 1) & (df_gat.y <= Y_DATA_MAX + 1)
].copy()

has_vgg_meas = (df_gat['gzz_meas'].notna().any() and (df_gat['gzz_meas'].abs() > 0.01).any())
df_gat['residual'] = df_gat['gzz_meas'] - df_gat['gzz_comp'] if has_vgg_meas else np.nan
print(f"  GAT: {len(df_gat)} titik")

print("Membaca file .blx ...")
blx_rows = []
with open(FILE_BLX) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            vals = [float(v) for v in line.split()]
            if len(vals) >= 8:
                blx_rows.append(vals[:8])
        except:
            continue

df_blx = pd.DataFrame(blx_rows, columns=['dx', 'dy', 'dz', 'xpos', 'ypos', 'zpos', 'fix_free', 'density'])
z_layers = sorted(df_blx.zpos.unique())
df_blx['layer'] = df_blx['zpos'].map({z: i + 1 for i, z in enumerate(z_layers)})
print(f"  BLX: {len(df_blx)} blok")

# ───────────────────────────────────────────────
# KOORDINAT
# ───────────────────────────────────────────────
R_EARTH = 6371.0
LAT_MEAN = np.radians((LAT_MIN + LAT_MAX) / 2)

def km_to_lon(x_km):
    return LON_MIN + np.degrees(x_km / (R_EARTH * np.cos(LAT_MEAN)))

def km_to_lat(y_km):
    return LAT_MIN + np.degrees(y_km / R_EARTH)

for df in [df_dat, df_gat]:
    df['lon'] = km_to_lon(df['x'])
    df['lat'] = km_to_lat(df['y'])
df_blx['lon'] = km_to_lon(df_blx['xpos'])
df_blx['lat'] = km_to_lat(df_blx['ypos'])

# ───────────────────────────────────────────────
# LOAD SLAB2
# ───────────────────────────────────────────────
print("Membaca data Slab2 ...")
try:
    slab_ds = xr.open_dataset('Banda_Arc_Slab2_Merged.grd')
    slab_ds = slab_ds.sel(x=slice(LON_MIN, LON_MAX), y=slice(LAT_MIN, LAT_MAX))
    slab_lon, slab_lat = np.meshgrid(slab_ds['x'].values, slab_ds['y'].values)
    slab_z = slab_ds['z'].values
    has_slab = True
    print("  Slab2 OK.")
except Exception as e:
    has_slab = False
    print(f"  [!] Slab2 gagal: {e}")

# ───────────────────────────────────────────────
# LOAD COASTLINE (dari shapefile lokal)
# ───────────────────────────────────────────────
COAST_SEGMENTS = []

def load_coastline_from_shp(shp_path, region):
    segs = []
    try:
        sf = shapefile.Reader(shp_path)
        for shape in sf.shapes():
            pts = np.array(shape.points)
            parts = list(shape.parts) + [len(pts)]
            for i in range(len(parts) - 1):
                seg = pts[parts[i]:parts[i+1]]
                if seg[:,0].max() < region[0] or seg[:,0].min() > region[1]:
                    continue
                if seg[:,1].max() < region[2] or seg[:,1].min() > region[3]:
                    continue
                segs.append(seg)
        return segs
    except Exception as e:
        print(f"  [!] Gagal baca shapefile {shp_path}: {e}")
        return []

shp_coast = "ne_shapefiles/ne_10m_coastline.shp"
shp_land   = "ne_shapefiles/ne_10m_land.shp"

if os.path.exists(shp_coast):
    COAST_SEGMENTS = load_coastline_from_shp(shp_coast, [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    print(f"  Coastline dari {shp_coast}: {len(COAST_SEGMENTS)} segmen")
elif os.path.exists(shp_land):
    COAST_SEGMENTS = load_coastline_from_shp(shp_land, [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    print(f"  Coastline dari {shp_land}: {len(COAST_SEGMENTS)} segmen")
else:
    print("  [!] File shapefile tidak ditemukan, coba download GeoJSON...")
    # fallback ke download GeoJSON
    try:
        _url = ("https://raw.githubusercontent.com/nvkelso/"
                "natural-earth-vector/master/geojson/ne_50m_coastline.geojson")
        with urllib.request.urlopen(_url, timeout=15) as resp:
            _gj = json.loads(resp.read())
        for feat in _gj['features']:
            geom = feat['geometry']
            segs_raw = ([geom['coordinates']] if geom['type'] == 'LineString'
                        else geom['coordinates'])
            for seg_raw in segs_raw:
                seg = np.array(seg_raw)
                if seg.ndim != 2 or seg.shape[1] < 2:
                    continue
                mask_c = ((seg[:, 0] >= LON_MIN - 0.5) & (seg[:, 0] <= LON_MAX + 0.5) &
                          (seg[:, 1] >= LAT_MIN - 0.5) & (seg[:, 1] <= LAT_MAX + 0.5))
                if mask_c.sum() > 2:
                    COAST_SEGMENTS.append(seg[mask_c])
        print(f"  Coastline NE-50m (download): {len(COAST_SEGMENTS)} segmen")
    except Exception as _ce:
        print(f"  [!] Coastline gagal dimuat dari manapun: {_ce}")

def draw_coastline(ax):
    for seg in COAST_SEGMENTS:
        ax.plot(seg[:, 0], seg[:, 1], color='black', lw=0.7, zorder=8, alpha=0.9)

# ───────────────────────────────────────────────
# FUNGSI UTILITAS
# ───────────────────────────────────────────────
def to_grid(df, xcol, ycol, vcol, nx=NX_DATA, ny=NY_DATA):
    xi = np.linspace(df[xcol].min(), df[xcol].max(), nx)
    yi = np.linspace(df[ycol].min(), df[ycol].max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)
    pts = df[[xcol, ycol]].values
    vals = df[vcol].values
    mask = np.isfinite(vals)
    Zi = griddata(pts[mask], vals[mask], (Xi, Yi), method='linear')
    return xi, yi, Xi, Yi, Zi

def format_map_ax(ax, lon_step=4, lat_step=2):
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(lon_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(lat_step))
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', linewidth=0.4, color='gray', alpha=0.5)

def add_slab_contours(ax, levels=np.arange(25, 175, 25), lw=2.0, alpha=0.8, label_fs=9):
    if not has_slab:
        return
    CS = ax.contour(slab_lon, slab_lat, np.abs(slab_z),
                    levels=levels, colors='black',
                    linestyles='dashed', linewidths=lw, alpha=alpha, zorder=10)
    clbls = ax.clabel(CS, inline=True, fontsize=label_fs, fmt='%d km', colors='black')
    for lbl in clbls:
        lbl.set_path_effects([pe.withStroke(linewidth=3, foreground='white'), pe.Normal()])

def add_trench_line(ax, lw=2.5):
    if not has_slab:
        return
    slab_abs = np.abs(slab_z)
    trench_lons, trench_lats = [], []
    for i, lon_val in enumerate(slab_ds['x'].values):
        col_data = slab_abs[:, i]
        vmask = np.isfinite(col_data)
        if vmask.sum() < 3:
            continue
        lat_vals = slab_ds['y'].values[vmask]
        min_idx = np.nanargmin(col_data[vmask])
        trench_lons.append(lon_val)
        trench_lats.append(lat_vals[min_idx])
    if len(trench_lons) > 5:
        sort_idx = np.argsort(trench_lons)
        t_lons = np.array(trench_lons)[sort_idx]
        t_lats = uniform_filter1d(np.array(trench_lats)[sort_idx], size=7)
        ax.plot(t_lons, t_lats, 'k-', linewidth=lw, zorder=10,
                path_effects=[pe.withStroke(linewidth=lw + 2, foreground='white'), pe.Normal()])
        mid = len(t_lons) // 2
        ax.text(t_lons[mid], t_lats[mid] + 0.3, 'Trench',
                fontsize=8, fontweight='bold', color='black', ha='center', va='bottom', zorder=10,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

def panel_label(ax, letter, x=0.02, y=0.97):
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

def colorbar_right(fig, ax, mappable, label, width="3%", pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=width, pad=pad)
    cb = fig.colorbar(mappable, cax=cax, extend='both')
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb

# ───────────────────────────────────────────────
# GLOBAL VARIABLES UNTUK DENSITAS
# ───────────────────────────────────────────────
layer_mean = df_blx.groupby('layer')['density'].mean().to_dict()
depth_labels = {
    1: "Oceanic water",
    2: "Sediment / Upper crust",
    3: "Lower crystalline crust",
    4: "Upper mantle",
    5: "Mantle",
    6: "Mantle",
    7: "Deep mantle",
    8: "Deep mantle"
}

# --- Kedalaman seragam untuk 8 layer ---
TOTAL_DEPTH = 125.0          # km
N_LAYERS = 8
THICKNESS = TOTAL_DEPTH / N_LAYERS   # 15.625 km
depth_edges = np.array([i * THICKNESS for i in range(N_LAYERS + 1)])
depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2

# --- Definisikan colormap densitas dan norm ---
# Ambil semua densitas dari semua layer untuk menentukan rentang
all_densities = df_blx['density'].values
DENS_VMIN = np.floor(np.percentile(all_densities, 10) * 20) / 20
DENS_VMAX = np.ceil(np.percentile(all_densities, 90) * 20) / 20
print(f"  Densitas global (10-90 pct): {DENS_VMIN:.3f} - {DENS_VMAX:.3f} g/cm3")

_dens_nodes = [
    (0.00, '#1a4fa0'),
    (0.30, '#6baed6'),
    (0.50, '#f5f5f5'),
    (0.70, '#f16913'),
    (1.00, '#8c0000'),
]
DENS_CMAP = LinearSegmentedColormap.from_list('dens_vivid', _dens_nodes, N=512)
BG_DENS = 2.84
norm_dens = TwoSlopeNorm(vmin=DENS_VMIN, vcenter=BG_DENS, vmax=DENS_VMAX)

# ═══════════════════════════════════════════════════════════
# FIGURE 1: PETA BOUGUER
# ═══════════════════════════════════════════════════════════
print("Membuat Figure 1: Peta Bouguer...")
bouguer_panels = [
    ('g_meas_corr', 'Anomali Bouguer Terkoreksi'),
    ('g_comp',      'Anomali Bouguer Kalkulasi (Forward Modelling)'),
    ('residual',    'Residual (Terkoreksi − Kalkulasi)'),
]
fig1, axes1 = plt.subplots(3, 1, figsize=(7, 14), gridspec_kw={'hspace': 0.35})
for idx, (col, title) in enumerate(bouguer_panels):
    ax = axes1[idx]
    _, _, Xi, Yi, Zi = to_grid(df_dat, 'lon', 'lat', col)
    finite = Zi[np.isfinite(Zi)]
    vabs = max(abs(np.nanpercentile(Zi, 1)), abs(np.nanpercentile(Zi, 99))) or 1e-5
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    cf = ax.contourf(Xi, Yi, Zi, levels=50, cmap='RdBu_r', norm=norm, extend='both')
    ax.contour(Xi, Yi, Zi, levels=10, colors='black', linewidths=0.25, alpha=0.3)
    add_slab_contours(ax)
    add_trench_line(ax)
    draw_coastline(ax)
    cb = colorbar_right(fig1, ax, cf, 'mGal')
    ax.set_title(title, fontsize=9, pad=4)
    format_map_ax(ax, lon_step=4, lat_step=2)
    panel_label(ax, chr(ord('a') + idx))
fig1.savefig("Figure1_Bouguer.png", dpi=300, bbox_inches='tight')
plt.close()
print("  -> Figure1_Bouguer.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 2: PETA VGG
# ═══════════════════════════════════════════════════════════
print("Membuat Figure 2: Peta VGG...")
if has_vgg_meas:
    vgg_panels = [
        ('gzz_meas', 'Vertical Gravity Gradient Terukur'),
        ('gzz_comp', 'Vertical Gravity Gradient Kalkulasi (Forward Modelling)'),
        ('residual', 'Residual (Terukur − Kalkulasi)'),
    ]
    vgg_df = df_gat
else:
    vgg_panels = [
        ('gzz_comp', 'Vertical Gravity Gradient Kalkulasi'),
        ('gzz_comp', 'Vertical Gravity Gradient Kalkulasi (copy)'),
        ('residual', 'Residual'),
    ]
    vgg_df = df_gat
fig2, axes2 = plt.subplots(3, 1, figsize=(7, 14), gridspec_kw={'hspace': 0.35})
for idx, (col, title) in enumerate(vgg_panels):
    ax = axes2[idx]
    _, _, Xi, Yi, Zi = to_grid(vgg_df, 'lon', 'lat', col)
    finite = Zi[np.isfinite(Zi)]
    if len(finite) == 0:
        ax.set_title(title + ' (no data)', fontsize=9)
        panel_label(ax, chr(ord('a') + idx))
        continue
    vabs = max(abs(np.nanpercentile(finite, 1)), abs(np.nanpercentile(finite, 99))) or 1e-5
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    cf = ax.contourf(Xi, Yi, Zi, levels=50, cmap='RdBu_r', norm=norm, extend='both')
    ax.contour(Xi, Yi, Zi, levels=10, colors='black', linewidths=0.25, alpha=0.3)
    add_slab_contours(ax)
    add_trench_line(ax)
    draw_coastline(ax)
    cb = colorbar_right(fig2, ax, cf, 'Eötvös')
    ax.set_title(title, fontsize=9, pad=4)
    format_map_ax(ax, lon_step=4, lat_step=2)
    panel_label(ax, chr(ord('a') + idx))
fig2.savefig("Figure2_VGG.png", dpi=300, bbox_inches='tight')
plt.close()
print("  -> Figure2_VGG.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 3: MODEL DENSITAS 3D (dengan kedalaman seragam)
# ═══════════════════════════════════════════════════════════
print("Membuat Figure 3: Model Densitas 3D...")

# Ambil data per layer (seperti sebelumnya)
layer_data = {}
for lyr in range(1, 9):
    sub = df_blx[df_blx['layer'] == lyr].copy()
    if len(sub) == 0:
        continue
    sub = sub[(sub.xpos >= X_DATA_MIN - 10) & (sub.xpos <= X_DATA_MAX + 10) &
              (sub.ypos >= Y_DATA_MIN - 10) & (sub.ypos <= Y_DATA_MAX + 10)]
    if len(sub) > 10:
        layer_data[lyr] = sub
show_layers = sorted(layer_data.keys())[:8]

fig3 = plt.figure(figsize=(18, 14))
gs3 = gridspec.GridSpec(3, 4, figure=fig3, height_ratios=[1, 1, 0.55], hspace=0.52, wspace=0.32)
for idx, lyr in enumerate(show_layers):
    row, col_idx = divmod(idx, 4)
    ax = fig3.add_subplot(gs3[row, col_idx])
    sub = layer_data[lyr]
    _, _, Xi, Yi, Zi = to_grid(sub, 'lon', 'lat', 'density', nx=60, ny=40)
    cf = ax.contourf(Xi, Yi, Zi, levels=30, cmap=DENS_CMAP, norm=norm_dens, extend='both')
    ax.contour(Xi, Yi, Zi, levels=6, colors='black', linewidths=0.25, alpha=0.3)
    if has_slab:
        z_top_slab = depth_centers[lyr-1]   # gunakan kedalaman tengah seragam
        slab_abs = np.abs(slab_z)
        all_slab_lvls = np.array([5, 10, 15, 25, 50, 75, 100, 125, 150, 200])
        closest = all_slab_lvls[np.argmin(np.abs(all_slab_lvls - z_top_slab))]
        try:
            CS_hl = ax.contour(slab_lon, slab_lat, slab_abs,
                               levels=[closest], colors='black',
                               linestyles='solid', linewidths=2.0, alpha=0.9, zorder=10)
            clbls = ax.clabel(CS_hl, inline=True, fontsize=8, fmt='%d km', colors='black')
            for lbl in clbls:
                lbl.set_path_effects([pe.withStroke(linewidth=3, foreground='white'), pe.Normal()])
        except Exception:
            pass
        add_trench_line(ax, lw=2.0)
    draw_coastline(ax)
    divider = make_axes_locatable(ax)
    cax_i = divider.append_axes("right", size="4%", pad=0.04)
    cbi = fig3.colorbar(cf, cax=cax_i, extend='both')
    cbi.ax.tick_params(labelsize=6)
    cbi.ax.axhline(BG_DENS, color='black', linewidth=1.0, linestyle='--', alpha=0.8)
    mean_rho = layer_mean.get(lyr, sub['density'].mean())
    # Kedalaman seragam
    zt = depth_edges[lyr-1]
    zb = depth_edges[lyr]
    geo_label = depth_labels.get(lyr, f"Layer {lyr}")
    panel_letter = chr(ord('a') + (idx % 4))
    ax.set_title(
        f"({panel_letter}) Layer {lyr}  |  {zt:.1f}-{zb:.1f} km\n"
        f"{geo_label}   mu = {mean_rho:.3f} g/cm3",
        fontsize=7.5, pad=3)
    format_map_ax(ax, lon_step=4, lat_step=2)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Lon (E)", fontsize=7)
    ax.set_ylabel("Lat", fontsize=7)

# Bar chart
ax_bar3 = fig3.add_subplot(gs3[2, :])
layers_bar = sorted(layer_mean.keys())
densities_bar = [layer_mean[l] for l in layers_bar]
labels_bar = [
    f"Layer {l}\n{depth_labels.get(l,'')}\n({depth_edges[l-1]:.1f}-{depth_edges[l]:.1f} km)"
    for l in layers_bar
]
colors_bar3 = [DENS_CMAP(norm_dens(d)) for d in densities_bar]
ax_bar3.bar(range(len(layers_bar)), densities_bar,
            color=colors_bar3, edgecolor='#333333', linewidth=0.6, width=0.6)
ax_bar3.set_xticks(range(len(layers_bar)))
ax_bar3.set_xticklabels(labels_bar, fontsize=7.5)
ax_bar3.set_ylabel("Mean density (g/cm3)", fontsize=9)
ax_bar3.set_title("(i)  Mean density per layer from 3D inversion", fontsize=9, fontweight='bold')
ax_bar3.axhline(BG_DENS, color='black', lw=1.2, ls='--', alpha=0.8,
                label=f'Background density = {BG_DENS} g/cm3')
ref_lines = {'Sediment (2.2)': 2.2, 'Avg. crust (2.75)': 2.75, 'Upper mantle (3.2)': 3.2}
ref_colors_bar = ['#4dac26', '#7b3294', '#d01c8b']
for (rlbl, rval), rcol in zip(ref_lines.items(), ref_colors_bar):
    ax_bar3.axhline(rval, color=rcol, lw=0.8, ls=':', alpha=0.7, label=rlbl)
ax_bar3.legend(fontsize=7.5, loc='upper right', framealpha=0.85, ncol=2)
ax_bar3.tick_params(labelsize=8)
ax_bar3.grid(True, linestyle=':', linewidth=0.5, alpha=0.4, axis='y')
ax_bar3.set_ylim(DENS_VMIN - 0.05, DENS_VMAX + 0.25)
for i, val in enumerate(densities_bar):
    ax_bar3.text(i, val + 0.03, f'{val:.3f}', ha='center', va='bottom',
                 fontsize=7.5, fontweight='bold')
fig3.savefig("Figure3_Density_Layers.png", dpi=300, bbox_inches='tight')
plt.close()
print("  -> Figure3_Density_Layers.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 4: VALIDASI SCATTER
# ═══════════════════════════════════════════════════════════
print("Membuat Figure 4: Validasi...")
fig4, axes4 = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={'wspace': 0.38})
fig4.suptitle("Validasi Model — Terkoreksi/Terukur vs Kalkulasi", fontsize=11, fontweight='bold')

def make_scatter_panel(ax, x, y, xlabel, ylabel, title, letter, unit):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=6, alpha=0.55, color='black', rasterized=True,
               linewidths=0, marker='o')
    ss_res = np.sum((x - y) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((x - y) ** 2))
    bias = np.mean(y - x)
    n_pts = len(x)
    stats_txt = (f"$R^2$  = {r2:.4f}\n"
                 f"RMSE = {rmse:.2f} {unit}\n"
                 f"Bias  = {bias:+.2f} {unit}\n"
                 f"$n$    = {n_pts:,}")
    ax.text(0.04, 0.97, stats_txt,
            transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
            family='monospace',
            bbox=dict(facecolor='white', edgecolor='#888888',
                      boxstyle='round,pad=0.4', alpha=0.93))
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(f"({letter})  {title}", fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8.5)
    ax.grid(True, linestyle=':', linewidth=0.45, alpha=0.55)
    ax.set_aspect('equal', adjustable='box')

make_scatter_panel(
    ax=axes4[0],
    x=df_dat['g_meas_corr'].values,
    y=df_dat['g_comp'].values,
    xlabel="Anomali Bouguer Terkoreksi (mGal)",
    ylabel="Anomali Bouguer Kalkulasi (mGal)",
    title="Anomali Bouguer",
    letter='a',
    unit='mGal'
)
if has_vgg_meas:
    make_scatter_panel(
        ax=axes4[1],
        x=df_gat['gzz_meas'].values,
        y=df_gat['gzz_comp'].values,
        xlabel="Vertical Gravity Gradient Terukur (Eötvös)",
        ylabel="Vertical Gravity Gradient Kalkulasi (Eötvös)",
        title="Vertical Gravity Gradient (VGG)",
        letter='b',
        unit='Eötvös'
    )
else:
    axes4[1].text(0.5, 0.5, 'VGG terukur tidak tersedia',
                  ha='center', va='center', transform=axes4[1].transAxes,
                  fontsize=10, color='gray')
    axes4[1].set_title("(b)  VGG", fontsize=10, fontweight='bold')
fig4.savefig("Figure4_Validasi.png", dpi=300, bbox_inches='tight')
plt.close()
print("  -> Figure4_Validasi.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 5: PROFIL BOUGUER + CROSS-SECTION DENSITAS (kedalaman seragam)
# ═══════════════════════════════════════════════════════════
print("Membuat Figure 5: Profil Bouguer & Cross-Section Densitas...")
fig5 = plt.figure(figsize=(20, 14))
gs5 = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.45, wspace=0.38,
                        width_ratios=[1.4, 1.4, 1.0])

# Panel lokasi
ax_loc = fig5.add_subplot(gs5[0, 2])
_, _, Xi_b, Yi_b, Zi_b = to_grid(df_dat, 'lon', 'lat', 'g_meas_corr')
vabs_loc = max(abs(np.nanpercentile(Zi_b, 2)), abs(np.nanpercentile(Zi_b, 98))) or 1e-5
norm_loc = TwoSlopeNorm(vmin=-vabs_loc, vcenter=0, vmax=vabs_loc)
ax_loc.contourf(Xi_b, Yi_b, Zi_b, levels=50, cmap='RdBu_r', norm=norm_loc, extend='both')
ax_loc.axhline(-6.5, color='gold', lw=1.5, ls='-', zorder=5,
               path_effects=[pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()])
ax_loc.axvline(126.0, color='limegreen', lw=1.5, ls='-', zorder=5,
               path_effects=[pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()])
ax_loc.text(LON_MAX - 0.3, -6.5 + 0.3, 'E–W', fontsize=7, fontweight='bold',
            color='gold', ha='right',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])
ax_loc.text(126.0 + 0.2, LAT_MAX - 0.5, 'N–S', fontsize=7, fontweight='bold',
            color='limegreen', ha='left',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])
ax_loc.plot(126.0, -6.5, 'w*', ms=8, zorder=7,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
if has_slab:
    CS_loc = ax_loc.contour(slab_lon, slab_lat, np.abs(slab_z),
                             levels=[25, 75, 125], colors='black',
                             linestyles='dashed', linewidths=1.5, alpha=0.8)
    ax_loc.clabel(CS_loc, inline=True, fontsize=7, fmt='%d km')
draw_coastline(ax_loc)
format_map_ax(ax_loc, lon_step=4, lat_step=2)
ax_loc.set_title("Lokasi Profil\n(pada peta Anomali Bouguer)", fontsize=8, fontweight='bold')

# Profil E-W
ax_ew = fig5.add_subplot(gs5[0, 0])
df_ew = df_dat[np.abs(df_dat['lat'] - (-6.5)) < 0.15].sort_values('lon')
if len(df_ew) > 5:
    ax_ew.plot(df_ew['lon'], df_ew['g_meas_corr'], '-', color='#2166ac',
               lw=1.3, label='Terkoreksi', zorder=3)
    ax_ew.plot(df_ew['lon'], df_ew['g_comp'], '-', color='#d6604d',
               lw=1.3, label='Kalkulasi', zorder=3)
    ax_ew.fill_between(df_ew['lon'], df_ew['g_meas_corr'], df_ew['g_comp'],
                       alpha=0.12, color='gray', label='Residual')
ax_ew.set_xlabel("Longitude (°E)", fontsize=9)
ax_ew.set_ylabel("Anomali Bouguer (mGal)", fontsize=9)
ax_ew.set_title("(a)  Profil E–W (lintang = −6,5°)", fontsize=9, fontweight='bold')
ax_ew.set_xlim(LON_MIN, LON_MAX)
ax_ew.legend(fontsize=8)
ax_ew.tick_params(labelsize=8)
ax_ew.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
ax_ew.xaxis.set_major_locator(ticker.MultipleLocator(4))

# Profil N-S
ax_ns = fig5.add_subplot(gs5[0, 1])
df_ns = df_dat[np.abs(df_dat['lon'] - 126.0) < 0.15].sort_values('lat')
if len(df_ns) > 5:
    ax_ns.plot(df_ns['lat'], df_ns['g_meas_corr'], '-', color='#2166ac',
               lw=1.3, label='Terkoreksi', zorder=3)
    ax_ns.plot(df_ns['lat'], df_ns['g_comp'], '-', color='#d6604d',
               lw=1.3, label='Kalkulasi', zorder=3)
    ax_ns.fill_between(df_ns['lat'], df_ns['g_meas_corr'], df_ns['g_comp'],
                       alpha=0.12, color='gray', label='Residual')
ax_ns.set_xlabel("Latitude (°)", fontsize=9)
ax_ns.set_ylabel("Anomali Bouguer (mGal)", fontsize=9)
ax_ns.set_title("(b)  Profil U–S (bujur = 126°T)", fontsize=9, fontweight='bold')
ax_ns.set_xlim(LAT_MIN, LAT_MAX)
ax_ns.legend(fontsize=8)
ax_ns.tick_params(labelsize=8)
ax_ns.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
ax_ns.xaxis.set_major_locator(ticker.MultipleLocator(2))

# ----------------------------------------------------------------------
# CROSS-SECTION E-W (dengan kedalaman seragam)
ax_cs_ew = fig5.add_subplot(gs5[1, 0])
lat_cs = -6.5
df_cs = df_blx[np.abs(df_blx['lat'] - lat_cs) < 0.5].sort_values(['xpos', 'zpos'])
if len(df_cs) > 10:
    lons_cs = sorted(df_cs['lon'].unique())
    layers_cs = sorted(df_cs['layer'].unique())
    Z_cs = np.full((len(layers_cs), len(lons_cs)), np.nan)
    for iz, lyr in enumerate(layers_cs):
        sub_l = df_cs[df_cs['layer'] == lyr]
        for ix, lv in enumerate(lons_cs):
            row = sub_l[np.abs(sub_l['lon'] - lv) < 0.2]
            if len(row) > 0:
                Z_cs[iz, ix] = row['density'].iloc[0]
    
    if len(lons_cs) > 1:
        dlon = (lons_cs[1] - lons_cs[0]) / 2
        lon_edges_cs = np.concatenate([[lons_cs[0] - dlon], lons_cs + dlon])
    else:
        lon_edges_cs = np.array([lons_cs[0] - 0.5, lons_cs[0] + 0.5])
    
    Lon_mesh, Dep_mesh = np.meshgrid(lon_edges_cs, depth_edges)
    cf_ew = ax_cs_ew.pcolormesh(Lon_mesh, Dep_mesh, Z_cs, cmap=DENS_CMAP, norm=norm_dens, shading='flat')
    ax_cs_ew.contour(Lon_mesh[:-1,:-1], Dep_mesh[:-1,:-1], Z_cs, levels=10, colors='black', linewidths=0.25, alpha=0.3)
    colorbar_right(fig5, ax_cs_ew, cf_ew, 'Density (g/cm³)', width="3%", pad=0.05)
    
    for iz, lyr in enumerate(layers_cs):
        y_mid = depth_centers[iz]
        ax_cs_ew.text(LON_MIN + 0.15, y_mid, depth_labels.get(lyr, f"L{lyr}"),
                      fontsize=5.5, color='white', fontweight='bold',
                      path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
    
    if has_slab:
        slab_ew_lons = slab_ds['x'].values
        slab_ew_lats = slab_ds['y'].values
        lat_idx = np.argmin(np.abs(slab_ew_lats - lat_cs))
        slab_depth_profile = np.abs(slab_z[lat_idx, :])
        valid_slab = np.isfinite(slab_depth_profile) & (slab_depth_profile <= 130)
        if valid_slab.sum() > 3:
            ax_cs_ew.plot(slab_ew_lons[valid_slab], slab_depth_profile[valid_slab],
                          'k--', lw=2.0, alpha=0.85, label='Slab2 geometry', zorder=10)
            ax_cs_ew.legend(fontsize=7, loc='lower right',
                            framealpha=0.85, edgecolor='#aaaaaa')
    
    ax_cs_ew.set_xlabel("Bujur (°T)", fontsize=9)
    ax_cs_ew.set_ylabel("Kedalaman (km)", fontsize=9)
    ax_cs_ew.set_title(f"(a)  Penampang Vertikal E–W (lintang = {lat_cs}°)", fontsize=9, fontweight='bold')
    ax_cs_ew.set_ylim(125, 0)
    ax_cs_ew.set_xlim(LON_MIN, LON_MAX)
    ax_cs_ew.tick_params(labelsize=8)
    ax_cs_ew.grid(True, linestyle=':', linewidth=0.4, color='white', alpha=0.25)
    ax_cs_ew.xaxis.set_major_locator(ticker.MultipleLocator(4))

# ═══════════════════════════════════════════════════════════
# FIGURE 5: PROFIL BOUGUER + CROSS-SECTION DENSITAS
# (kembali ke metode awal untuk penampang, label Indonesia)
# ═══════════════════════════════════════════════════════════
print("Membuat Figure 5: Profil Bouguer & Cross-Section Densitas...")
fig5 = plt.figure(figsize=(20, 14))
gs5 = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.45, wspace=0.38,
                        width_ratios=[1.4, 1.4, 1.0])

# ------------------------------------------------------------------
# Panel lokasi (kiri atas ke-3)
# ------------------------------------------------------------------
ax_loc = fig5.add_subplot(gs5[0, 2])
_, _, Xi_b, Yi_b, Zi_b = to_grid(df_dat, 'lon', 'lat', 'g_meas_corr')
vabs_loc = max(abs(np.nanpercentile(Zi_b, 2)), abs(np.nanpercentile(Zi_b, 98))) or 1e-5
norm_loc = TwoSlopeNorm(vmin=-vabs_loc, vcenter=0, vmax=vabs_loc)
ax_loc.contourf(Xi_b, Yi_b, Zi_b, levels=50, cmap='RdBu_r', norm=norm_loc, extend='both')
ax_loc.axhline(-6.5, color='gold', lw=1.5, ls='-', zorder=5,
               path_effects=[pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()])
ax_loc.axvline(126.0, color='limegreen', lw=1.5, ls='-', zorder=5,
               path_effects=[pe.withStroke(linewidth=2.5, foreground='black'), pe.Normal()])
ax_loc.text(LON_MAX - 0.3, -6.5 + 0.3, 'E–W', fontsize=7, fontweight='bold',
            color='gold', ha='right',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])
ax_loc.text(126.0 + 0.2, LAT_MAX - 0.5, 'N–S', fontsize=7, fontweight='bold',
            color='limegreen', ha='left',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])
ax_loc.plot(126.0, -6.5, 'w*', ms=8, zorder=7,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
if has_slab:
    CS_loc = ax_loc.contour(slab_lon, slab_lat, np.abs(slab_z),
                             levels=[25, 75, 125], colors='black',
                             linestyles='dashed', linewidths=1.5, alpha=0.8)
    ax_loc.clabel(CS_loc, inline=True, fontsize=7, fmt='%d km')
draw_coastline(ax_loc)
format_map_ax(ax_loc, lon_step=4, lat_step=2)
ax_loc.set_title("Lokasi Profil\n(pada peta Anomali Bouguer)", fontsize=8, fontweight='bold')

# ------------------------------------------------------------------
# Profil E-W (Bouguer)
# ------------------------------------------------------------------
ax_ew = fig5.add_subplot(gs5[0, 0])
df_ew = df_dat[np.abs(df_dat['lat'] - (-6.5)) < 0.15].sort_values('lon')
if len(df_ew) > 5:
    ax_ew.plot(df_ew['lon'], df_ew['g_meas_corr'], '-', color='#2166ac',
               lw=1.3, label='Terkoreksi', zorder=3)
    ax_ew.plot(df_ew['lon'], df_ew['g_comp'], '-', color='#d6604d',
               lw=1.3, label='Kalkulasi', zorder=3)
    ax_ew.fill_between(df_ew['lon'], df_ew['g_meas_corr'], df_ew['g_comp'],
                       alpha=0.12, color='gray', label='Residual')
ax_ew.set_xlabel("Longitude (°E)", fontsize=9)
ax_ew.set_ylabel("Anomali Bouguer (mGal)", fontsize=9)
ax_ew.set_title("(a)  Profil E–W (lintang = −6,5°)", fontsize=9, fontweight='bold')
ax_ew.set_xlim(LON_MIN, LON_MAX)
ax_ew.legend(fontsize=8)
ax_ew.tick_params(labelsize=8)
ax_ew.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
ax_ew.xaxis.set_major_locator(ticker.MultipleLocator(4))

# ------------------------------------------------------------------
# Profil N-S (Bouguer)
# ------------------------------------------------------------------
ax_ns = fig5.add_subplot(gs5[0, 1])
df_ns = df_dat[np.abs(df_dat['lon'] - 126.0) < 0.15].sort_values('lat')
if len(df_ns) > 5:
    ax_ns.plot(df_ns['lat'], df_ns['g_meas_corr'], '-', color='#2166ac',
               lw=1.3, label='Terkoreksi', zorder=3)
    ax_ns.plot(df_ns['lat'], df_ns['g_comp'], '-', color='#d6604d',
               lw=1.3, label='Kalkulasi', zorder=3)
    ax_ns.fill_between(df_ns['lat'], df_ns['g_meas_corr'], df_ns['g_comp'],
                       alpha=0.12, color='gray', label='Residual')
ax_ns.set_xlabel("Latitude (°)", fontsize=9)
ax_ns.set_ylabel("Anomali Bouguer (mGal)", fontsize=9)
ax_ns.set_title("(b)  Profil U–S (bujur = 126°T)", fontsize=9, fontweight='bold')
ax_ns.set_xlim(LAT_MIN, LAT_MAX)
ax_ns.legend(fontsize=8)
ax_ns.tick_params(labelsize=8)
ax_ns.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
ax_ns.xaxis.set_major_locator(ticker.MultipleLocator(2))

# ------------------------------------------------------------------
# Cross-section E-W (metode awal)
# ------------------------------------------------------------------
ax_cs_ew = fig5.add_subplot(gs5[1, 0])
lat_cs = -6.5
df_cs = df_blx[np.abs(df_blx['lat'] - lat_cs) < 0.5].sort_values(['xpos', 'zpos'])
if len(df_cs) > 10:
    lons_cs = sorted(df_cs['lon'].unique())
    layers_cs = sorted(df_cs['layer'].unique())
    Z_cs = np.full((len(layers_cs), len(lons_cs)), np.nan)
    z_depths_raw = [df_cs[df_cs['layer'] == lyr]['zpos'].iloc[0] for lyr in layers_cs]
    z_min_r, z_max_r = min(z_depths_raw), max(z_depths_raw)
    z_depths = [((z - z_min_r) / (z_max_r - z_min_r)) * 125 for z in z_depths_raw]

    for iz, lyr in enumerate(layers_cs):
        sub_l = df_cs[df_cs['layer'] == lyr]
        for ix, lv in enumerate(lons_cs):
            row = sub_l[np.abs(sub_l['lon'] - lv) < 0.2]
            if len(row) > 0:
                Z_cs[iz, ix] = row['density'].iloc[0]

    Lon_m, Dep_m = np.meshgrid(lons_cs, z_depths)
    cf_ew = ax_cs_ew.contourf(Lon_m, Dep_m, Z_cs, levels=25, cmap=DENS_CMAP,
                              norm=norm_dens, extend='both')
    ax_cs_ew.contour(Lon_m, Dep_m, Z_cs, levels=10, colors='black', linewidths=0.25, alpha=0.3)
    colorbar_right(fig5, ax_cs_ew, cf_ew, 'Density (g/cm³)', width="3%", pad=0.05)

    for iz, (lyr, zd) in enumerate(zip(layers_cs, z_depths)):
        ax_cs_ew.text(LON_MIN + 0.15, zd + 1.5, depth_labels.get(lyr, f"L{lyr}"),
                     fontsize=5.5, color='white', fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    if has_slab:
        slab_ew_lons = slab_ds['x'].values
        slab_ew_lats = slab_ds['y'].values
        lat_idx = np.argmin(np.abs(slab_ew_lats - lat_cs))
        slab_depth_profile = np.abs(slab_z[lat_idx, :])
        valid_slab = np.isfinite(slab_depth_profile) & (slab_depth_profile <= 130)
        if valid_slab.sum() > 3:
            slab_remap = ((slab_depth_profile[valid_slab] - z_min_r) / (z_max_r - z_min_r)) * 125
            ax_cs_ew.plot(slab_ew_lons[valid_slab], slab_remap,
                          'k--', lw=2.0, alpha=0.85, label='Slab2 geometry', zorder=10)
            ax_cs_ew.legend(fontsize=7, loc='lower right',
                            framealpha=0.85, edgecolor='#aaaaaa')

    ax_cs_ew.set_xlabel("Bujur (°T)", fontsize=9)
    ax_cs_ew.set_ylabel("Kedalaman (km)", fontsize=9)
    ax_cs_ew.set_title(f"(a)  Penampang Vertikal E–W (lintang = {lat_cs}°)", fontsize=9, fontweight='bold')
    ax_cs_ew.set_ylim(125, 0)
    ax_cs_ew.set_xlim(LON_MIN, LON_MAX)
    ax_cs_ew.tick_params(labelsize=8)
    ax_cs_ew.grid(True, linestyle=':', linewidth=0.4, color='white', alpha=0.25)
    ax_cs_ew.xaxis.set_major_locator(ticker.MultipleLocator(4))

# ------------------------------------------------------------------
# Cross-section N-S (metode awal)
# ------------------------------------------------------------------
ax_cs_ns = fig5.add_subplot(gs5[1, 1])
lon_cs = 126.0
df_cs_ns = df_blx[np.abs(df_blx['lon'] - lon_cs) < 0.5].sort_values(['ypos', 'zpos'])
if len(df_cs_ns) > 10:
    lats_cs = sorted(df_cs_ns['lat'].unique())
    layers_cs2 = sorted(df_cs_ns['layer'].unique())
    Z_cs2 = np.full((len(layers_cs2), len(lats_cs)), np.nan)
    z_depths_raw2 = [df_cs_ns[df_cs_ns['layer'] == lyr]['zpos'].iloc[0] for lyr in layers_cs2]
    z_min_r2, z_max_r2 = min(z_depths_raw2), max(z_depths_raw2)
    z_depths2 = [((z - z_min_r2) / (z_max_r2 - z_min_r2)) * 125 for z in z_depths_raw2]

    for iz, lyr in enumerate(layers_cs2):
        sub_l = df_cs_ns[df_cs_ns['layer'] == lyr]
        for iy, lv in enumerate(lats_cs):
            row = sub_l[np.abs(sub_l['lat'] - lv) < 0.2]
            if len(row) > 0:
                Z_cs2[iz, iy] = row['density'].iloc[0]

    Lat_m, Dep_m2 = np.meshgrid(lats_cs, z_depths2)
    cf_ns = ax_cs_ns.contourf(Lat_m, Dep_m2, Z_cs2, levels=25, cmap=DENS_CMAP,
                              norm=norm_dens, extend='both')
    ax_cs_ns.contour(Lat_m, Dep_m2, Z_cs2, levels=10, colors='black', linewidths=0.25, alpha=0.3)
    colorbar_right(fig5, ax_cs_ns, cf_ns, 'Density (g/cm³)', width="3%", pad=0.05)

    for iz, (lyr, zd) in enumerate(zip(layers_cs2, z_depths2)):
        ax_cs_ns.text(LAT_MIN + 0.15, zd + 1.5, depth_labels.get(lyr, f"L{lyr}"),
                     fontsize=5.5, color='white', fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    if has_slab:
        slab_ns_lons = slab_ds['x'].values
        slab_ns_lats = slab_ds['y'].values
        lon_idx = np.argmin(np.abs(slab_ns_lons - lon_cs))
        slab_depth_ns = np.abs(slab_z[:, lon_idx])
        valid_ns = np.isfinite(slab_depth_ns) & (slab_depth_ns <= 130)
        if valid_ns.sum() > 3:
            slab_remap_ns = ((slab_depth_ns[valid_ns] - z_min_r2) / (z_max_r2 - z_min_r2)) * 125
            ax_cs_ns.plot(slab_ns_lats[valid_ns], slab_remap_ns,
                          'k--', lw=2.0, alpha=0.85, label='Slab2 geometry', zorder=10)
            ax_cs_ns.legend(fontsize=7, loc='lower right',
                            framealpha=0.85, edgecolor='#aaaaaa')

    ax_cs_ns.set_xlabel("Lintang (°)", fontsize=9)
    ax_cs_ns.set_ylabel("Kedalaman (km)", fontsize=9)
    ax_cs_ns.set_title(f"(b)  Penampang Vertikal U–S (bujur = {lon_cs}°T)", fontsize=9, fontweight='bold')
    ax_cs_ns.set_ylim(125, 0)
    ax_cs_ns.set_xlim(LAT_MIN, LAT_MAX)
    ax_cs_ns.tick_params(labelsize=8)
    ax_cs_ns.grid(True, linestyle=':', linewidth=0.4, color='white', alpha=0.25)
    ax_cs_ns.xaxis.set_major_locator(ticker.MultipleLocator(2))

# ------------------------------------------------------------------
# Kosongkan panel (2,2)
# ------------------------------------------------------------------
fig5.add_subplot(gs5[1, 2]).axis('off')
draw_coastline(ax_cs_ew)
draw_coastline(ax_cs_ns)

fig5.savefig("Figure5_Profil_CrossSection.png", dpi=300, bbox_inches='tight')
plt.close()
print("  -> Figure5_Profil_CrossSection.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 6: PROFIL VGG
# ═══════════════════════════════════════════════════════════
if has_vgg_meas:
    print("Membuat Figure 6: VGG Profiles...")
    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 5))
    fig6.suptitle("Vertical Gravity Gradient (VGG) — Terukur vs Kalkulasi", fontsize=12, fontweight='bold')

    ax_ew_vgg = axes6[0]
    df_ew_v = df_gat[np.abs(df_gat['lat'] - (-6.5)) < 0.15].sort_values('lon')
    if len(df_ew_v) > 5:
        ax_ew_vgg.plot(df_ew_v['lon'], df_ew_v['gzz_meas'], '-o', color='#2166ac', ms=2, lw=1.3, label='Terukur')
        ax_ew_vgg.plot(df_ew_v['lon'], df_ew_v['gzz_comp'], '-s', color='#d6604d', ms=2, lw=1.3, label='Kalkulasi')
        ax_ew_vgg.fill_between(df_ew_v['lon'], df_ew_v['gzz_meas'], df_ew_v['gzz_comp'],
                               alpha=0.12, color='gray', label='Residual')
    ax_ew_vgg.set_xlabel("Longitude (°E)", fontsize=10)
    ax_ew_vgg.set_ylabel("VGG (Eötvös)", fontsize=10)
    ax_ew_vgg.set_title("(a)  Profil E–W (lintang = −6,5°)", fontsize=10, fontweight='bold')
    ax_ew_vgg.set_xlim(LON_MIN, LON_MAX)
    ax_ew_vgg.legend(fontsize=8)
    ax_ew_vgg.grid(True, linestyle=':', alpha=0.5)

    ax_ns_vgg = axes6[1]
    df_ns_v = df_gat[np.abs(df_gat['lon'] - 126.0) < 0.15].sort_values('lat')
    if len(df_ns_v) > 5:
        ax_ns_vgg.plot(df_ns_v['lat'], df_ns_v['gzz_meas'], '-o', color='#2166ac', ms=2, lw=1.3, label='Terukur')
        ax_ns_vgg.plot(df_ns_v['lat'], df_ns_v['gzz_comp'], '-s', color='#d6604d', ms=2, lw=1.3, label='Kalkulasi')
        ax_ns_vgg.fill_between(df_ns_v['lat'], df_ns_v['gzz_meas'], df_ns_v['gzz_comp'],
                               alpha=0.12, color='gray', label='Residual')
    ax_ns_vgg.set_xlabel("Lintang (°)", fontsize=10)
    ax_ns_vgg.set_ylabel("VGG (Eötvös)", fontsize=10)
    ax_ns_vgg.set_title("(b)  Profil U–S (bujur = 126°T)", fontsize=10, fontweight='bold')
    ax_ns_vgg.set_xlim(LAT_MIN, LAT_MAX)
    ax_ns_vgg.legend(fontsize=8)
    ax_ns_vgg.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig6.savefig("Figure6_VGG_Profiles.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Figure6_VGG_Profiles.png")
else:
    print("  Figure6 dilewati: VGG terukur tidak tersedia")

# ═══════════════════════════════════════════════════════════
# EKSPOR 1 FILE CSV RINGKASAN (tidak berubah)
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MEMBUAT 1 FILE CSV RINGKASAN")
print("="*60)

LAT_CS = -6.5
LON_CS = 126.0
rows = []

# 1. Profil Bouguer E-W
df_ew_bou = df_dat[np.abs(df_dat['lat'] - LAT_CS) < 0.15]
if len(df_ew_bou) > 0:
    rows.append({
        'Kategori': 'Bouguer E-W',
        'Koordinat': f'Lintang {LAT_CS}°',
        'Jenis': 'Terkoreksi',
        'Jumlah Data': len(df_ew_bou),
        'Min (mGal)': df_ew_bou['g_meas_corr'].min(),
        'Max (mGal)': df_ew_bou['g_meas_corr'].max(),
        'Mean (mGal)': df_ew_bou['g_meas_corr'].mean(),
        'Std (mGal)': df_ew_bou['g_meas_corr'].std()
    })
    rows.append({
        'Kategori': 'Bouguer E-W',
        'Koordinat': f'Lintang {LAT_CS}°',
        'Jenis': 'Kalkulasi',
        'Jumlah Data': len(df_ew_bou),
        'Min (mGal)': df_ew_bou['g_comp'].min(),
        'Max (mGal)': df_ew_bou['g_comp'].max(),
        'Mean (mGal)': df_ew_bou['g_comp'].mean(),
        'Std (mGal)': df_ew_bou['g_comp'].std()
    })

# 2. Profil Bouguer N-S
df_ns_bou = df_dat[np.abs(df_dat['lon'] - LON_CS) < 0.15]
if len(df_ns_bou) > 0:
    rows.append({
        'Kategori': 'Bouguer N-S',
        'Koordinat': f'Bujur {LON_CS}°E',
        'Jenis': 'Terkoreksi',
        'Jumlah Data': len(df_ns_bou),
        'Min (mGal)': df_ns_bou['g_meas_corr'].min(),
        'Max (mGal)': df_ns_bou['g_meas_corr'].max(),
        'Mean (mGal)': df_ns_bou['g_meas_corr'].mean(),
        'Std (mGal)': df_ns_bou['g_meas_corr'].std()
    })
    rows.append({
        'Kategori': 'Bouguer N-S',
        'Koordinat': f'Bujur {LON_CS}°E',
        'Jenis': 'Kalkulasi',
        'Jumlah Data': len(df_ns_bou),
        'Min (mGal)': df_ns_bou['g_comp'].min(),
        'Max (mGal)': df_ns_bou['g_comp'].max(),
        'Mean (mGal)': df_ns_bou['g_comp'].mean(),
        'Std (mGal)': df_ns_bou['g_comp'].std()
    })

# 3. Profil VGG E-W
if has_vgg_meas:
    df_ew_vgg = df_gat[np.abs(df_gat['lat'] - LAT_CS) < 0.15]
    if len(df_ew_vgg) > 0:
        rows.append({
            'Kategori': 'VGG E-W',
            'Koordinat': f'Lintang {LAT_CS}°',
            'Jenis': 'Terukur',
            'Jumlah Data': len(df_ew_vgg),
            'Min (Eötvös)': df_ew_vgg['gzz_meas'].min(),
            'Max (Eötvös)': df_ew_vgg['gzz_meas'].max(),
            'Mean (Eötvös)': df_ew_vgg['gzz_meas'].mean(),
            'Std (Eötvös)': df_ew_vgg['gzz_meas'].std()
        })
        rows.append({
            'Kategori': 'VGG E-W',
            'Koordinat': f'Lintang {LAT_CS}°',
            'Jenis': 'Kalkulasi',
            'Jumlah Data': len(df_ew_vgg),
            'Min (Eötvös)': df_ew_vgg['gzz_comp'].min(),
            'Max (Eötvös)': df_ew_vgg['gzz_comp'].max(),
            'Mean (Eötvös)': df_ew_vgg['gzz_comp'].mean(),
            'Std (Eötvös)': df_ew_vgg['gzz_comp'].std()
        })

# 4. Profil VGG N-S
if has_vgg_meas:
    df_ns_vgg = df_gat[np.abs(df_gat['lon'] - LON_CS) < 0.15]
    if len(df_ns_vgg) > 0:
        rows.append({
            'Kategori': 'VGG N-S',
            'Koordinat': f'Bujur {LON_CS}°E',
            'Jenis': 'Terukur',
            'Jumlah Data': len(df_ns_vgg),
            'Min (Eötvös)': df_ns_vgg['gzz_meas'].min(),
            'Max (Eötvös)': df_ns_vgg['gzz_meas'].max(),
            'Mean (Eötvös)': df_ns_vgg['gzz_meas'].mean(),
            'Std (Eötvös)': df_ns_vgg['gzz_meas'].std()
        })
        rows.append({
            'Kategori': 'VGG N-S',
            'Koordinat': f'Bujur {LON_CS}°E',
            'Jenis': 'Kalkulasi',
            'Jumlah Data': len(df_ns_vgg),
            'Min (Eötvös)': df_ns_vgg['gzz_comp'].min(),
            'Max (Eötvös)': df_ns_vgg['gzz_comp'].max(),
            'Mean (Eötvös)': df_ns_vgg['gzz_comp'].mean(),
            'Std (Eötvös)': df_ns_vgg['gzz_comp'].std()
        })

# 5. Penampang densitas E-W (per layer)
df_cs_ew = df_blx[np.abs(df_blx['lat'] - LAT_CS) < 0.5]
if len(df_cs_ew) > 0:
    for lyr in sorted(df_cs_ew['layer'].unique()):
        sub = df_cs_ew[df_cs_ew['layer'] == lyr]
        rows.append({
            'Kategori': f'Densitas E-W (Layer {lyr})',
            'Koordinat': f'Lintang {LAT_CS}°',
            'Jenis': f'Layer {lyr}',
            'Jumlah Data': len(sub),
            'Min (g/cm³)': sub['density'].min(),
            'Max (g/cm³)': sub['density'].max(),
            'Mean (g/cm³)': sub['density'].mean(),
            'Std (g/cm³)': sub['density'].std()
        })

# 6. Penampang densitas N-S (per layer)
df_cs_ns = df_blx[np.abs(df_blx['lon'] - LON_CS) < 0.5]
if len(df_cs_ns) > 0:
    for lyr in sorted(df_cs_ns['layer'].unique()):
        sub = df_cs_ns[df_cs_ns['layer'] == lyr]
        rows.append({
            'Kategori': f'Densitas N-S (Layer {lyr})',
            'Koordinat': f'Bujur {LON_CS}°E',
            'Jenis': f'Layer {lyr}',
            'Jumlah Data': len(sub),
            'Min (g/cm³)': sub['density'].min(),
            'Max (g/cm³)': sub['density'].max(),
            'Mean (g/cm³)': sub['density'].mean(),
            'Std (g/cm³)': sub['density'].std()
        })

# 7. Statistik global per layer
for lyr in sorted(df_blx['layer'].unique()):
    sub = df_blx[df_blx['layer'] == lyr]
    rows.append({
        'Kategori': 'Densitas Global',
        'Koordinat': 'Seluruh domain',
        'Jenis': f'Layer {lyr}',
        'Jumlah Data': len(sub),
        'Min (g/cm³)': sub['density'].min(),
        'Max (g/cm³)': sub['density'].max(),
        'Mean (g/cm³)': sub['density'].mean(),
        'Std (g/cm³)': sub['density'].std()
    })

df_summary = pd.DataFrame(rows)
df_summary.to_csv('summary_all.csv', index=False)
print("  -> summary_all.csv (semua ringkasan dalam 1 file)")
print(f"  Total baris: {len(df_summary)}")
print("\n" + "="*60)
print("  EKSPOR SELESAI")
print("="*60)

# ───────────────────────────────────────────────
# PARSER .OUT (tidak berubah)
# ───────────────────────────────────────────────
def parse_grablox_out(filepath):
    iterations = []
    summary = {}
    with open(filepath, 'r', errors='replace') as f:
        lines = f.readlines()
    iter_block = {}
    for line in lines:
        line_s = line.strip()
        m_iter = re.search(r'[Ii]teration\s*[:\s#]*(\d+)', line_s)
        if m_iter:
            if iter_block:
                iterations.append(iter_block.copy())
            iter_block = {'iteration': int(m_iter.group(1))}
        m_drms = re.search(r'[Dd]ata\s+RMS\s*[=:]\s*([\d.eE+\-]+)', line_s)
        if m_drms:
            iter_block['data_rms'] = float(m_drms.group(1))
        m_grms = re.search(r'[Gg]rad(?:ient)?\s+RMS\s*[=:]\s*([\d.eE+\-]+)', line_s)
        if m_grms:
            iter_block['grad_rms'] = float(m_grms.group(1))
        m_mrms = re.search(r'[Mm]odel\s+RMS\s*[=:]\s*([\d.eE+\-]+)', line_s)
        if m_mrms:
            iter_block['model_rms'] = float(m_mrms.group(1))
        m_lam = re.search(r'[Ll]ambda\s*[=:]\s*([\d.eE+\-]+)', line_s)
        if m_lam:
            iter_block['lambda'] = float(m_lam.group(1))
        m_rmse_b = re.search(r'RMSE.*[Bb]ouguer.*[=:]\s*([\d.eE+\-]+)', line_s)
        if m_rmse_b:
            summary['rmse_bouguer'] = float(m_rmse_b.group(1))
        m_rmse_v = re.search(r'RMSE.*(?:VGG|Gzz|gzz).*[=:]\s*([\d.eE+\-]+)', line_s)
        if m_rmse_v:
            summary['rmse_vgg'] = float(m_rmse_v.group(1))
        m_r2 = re.search(r'[Rr](?:2|²|\^2|_squared)\s*[=:]\s*([\d.eE+\-]+)', line_s)
        if m_r2:
            summary.setdefault('r2_list', []).append(float(m_r2.group(1)))
        m_mean = re.search(r'[Mm]ean\s+[Rr]esidual\s*[=:]\s*([\-\d.eE+]+)', line_s)
        if m_mean:
            summary['mean_residual'] = float(m_mean.group(1))
        m_std = re.search(r'[Ss]td\s*(?:[Dd]ev)?\s*[Rr]esidual\s*[=:]\s*([\d.eE+\-]+)', line_s)
        if m_std:
            summary['std_residual'] = float(m_std.group(1))
        m_npts = re.search(r'[Nn](?:umber)?\s*(?:of)?\s*(?:data\s*)?[Pp]oints?\s*[=:]\s*(\d+)', line_s)
        if m_npts:
            summary['n_data'] = int(m_npts.group(1))
    if iter_block:
        iterations.append(iter_block)
    df_iter = pd.DataFrame(iterations) if iterations else pd.DataFrame()
    if not df_iter.empty:
        last = df_iter.iloc[-1]
        for col in ['data_rms', 'grad_rms', 'model_rms', 'lambda']:
            if col in last and pd.notna(last[col]):
                summary[f'final_{col}'] = last[col]
        summary['n_iterations'] = len(df_iter)
    return summary, df_iter

def print_out_summary(summary, df_iter, filepath):
    print(f"\n{'═'*55}")
    print(f"  RINGKASAN KONVERGENSI — {filepath}")
    print(f"{'═'*55}")
    if 'n_iterations' in summary:
        print(f"  Jumlah iterasi       : {summary['n_iterations']}")
    if 'n_data' in summary:
        print(f"  Jumlah data point    : {summary['n_data']}")
    if 'final_data_rms' in summary:
        print(f"  Data RMS (akhir)     : {summary['final_data_rms']:.4f}")
    if 'final_grad_rms' in summary:
        print(f"  Grad RMS (akhir)     : {summary['final_grad_rms']:.6f}")
    if 'final_model_rms' in summary:
        print(f"  Model RMS (akhir)    : {summary['final_model_rms']:.4f}")
    if 'final_lambda' in summary:
        print(f"  Lambda (akhir)       : {summary['final_lambda']:.4e}")
    if 'rmse_bouguer' in summary:
        print(f"  RMSE Bouguer         : {summary['rmse_bouguer']:.4f} mGal")
    if 'rmse_vgg' in summary:
        print(f"  RMSE VGG             : {summary['rmse_vgg']:.4f} Eötvös")
    if 'r2_list' in summary:
        for val in summary['r2_list']:
            print(f"  R²                   : {val:.4f}")
    if 'mean_residual' in summary:
        print(f"  Mean residual        : {summary['mean_residual']:+.4f}")
    if 'std_residual' in summary:
        print(f"  Std residual         : {summary['std_residual']:.4f}")
    print(f"{'─'*55}")
    if not df_iter.empty and len(df_iter) > 2:
        rms_cols = [c for c in ['data_rms', 'grad_rms', 'model_rms'] if c in df_iter.columns]
        if rms_cols:
            fig_cv, ax_cv = plt.subplots(figsize=(7, 4))
            colors_cv = ['#2166ac', '#d73027', '#4dac26']
            labels_cv = {'data_rms': 'Data RMS', 'grad_rms': 'Grad RMS', 'model_rms': 'Model RMS'}
            for col, col_color in zip(rms_cols, colors_cv):
                sub = df_iter[['iteration', col]].dropna()
                if len(sub) > 1:
                    ax_cv.semilogy(sub['iteration'], sub[col], '-o',
                                   color=col_color, ms=3, lw=1.5,
                                   label=labels_cv.get(col, col))
            ax_cv.set_xlabel("Iteration", fontsize=10)
            ax_cv.set_ylabel("RMS (log scale)", fontsize=10)
            ax_cv.set_title("GRABlox2 Convergence — RMS per iteration",
                            fontsize=10, fontweight='bold')
            ax_cv.legend(fontsize=9)
            ax_cv.grid(True, linestyle=':', alpha=0.5)
            ax_cv.tick_params(labelsize=9)
            out_name = filepath.replace('.out', '').replace('.OUT', '') + '_convergence.png'
            fig_cv.savefig(out_name, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  Plot konvergensi     : {out_name}")

out_files = glob.glob("*.out") + glob.glob("*.OUT")
if out_files:
    for out_fp in sorted(out_files):
        try:
            summ, df_it = parse_grablox_out(out_fp)
            print_out_summary(summ, df_it, out_fp)
        except Exception as e:
            print(f"  [!] Gagal baca {out_fp}: {e}")
else:
    print("\n[INFO] Tidak ada file .out ditemukan di direktori ini.")