import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import griddata
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter
from scipy.signal.windows import tukey
from matplotlib.colors import TwoSlopeNorm

# =============================================================================
# PARAMETER
# =============================================================================
SIGMA_BOUGUER   = 1.0
K_CUT_FACTOR    = 0.3
CLIP_VGG        = True
VGG_CLIP_MIN, VGG_CLIP_MAX = -150, 150

# =============================================================================
# 1. BACA DATA
# =============================================================================
print("1. Membaca dan Menyiapkan Data...")
df_marine = pd.read_csv("marine_bouguer_anomaly_result.csv")[['lon', 'lat', 'DeltaGB_mGal']]
df_land   = pd.read_csv("gravity_anomaly_result.csv")[['Longitude', 'Latitude', 'DeltaGB_mGal']]
df_land.rename(columns={'Longitude': 'lon', 'Latitude': 'lat'}, inplace=True)

# =============================================================================
# 2. GRID 5 MENIT (193 x 109)
# =============================================================================
lon_grid = np.linspace(118.04166667, 134.04166667, 193)
lat_grid = np.linspace(-10.95833333, -1.95833333, 109)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# =============================================================================
# 3. GABUNG DATA (PRIORITAS LAUT)
# =============================================================================
df_marine['lon_rnd'] = df_marine['lon'].round(3)
df_marine['lat_rnd'] = df_marine['lat'].round(3)
df_land['lon_rnd']   = df_land['lon'].round(3)
df_land['lat_rnd']   = df_land['lat'].round(3)

df_combined = pd.concat([df_marine, df_land], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=['lon_rnd', 'lat_rnd'], keep='first')
print(f"Total data tergabung: {len(df_combined)} titik.")

# =============================================================================
# 4. INTERPOLASI CUBIC + FALLBACK NEAREST
# =============================================================================
print("2. Interpolasi grid 5-menit...")
points = df_combined[['lon', 'lat']].values
values = df_combined['DeltaGB_mGal'].values

bouguer_cubic = griddata(points, values, (lon_mesh, lat_mesh), method='cubic')
mask_nan = np.isnan(bouguer_cubic)
if mask_nan.any():
    bouguer_nearest = griddata(points, values, (lon_mesh, lat_mesh), method='nearest')
    bouguer_2d = bouguer_cubic.copy()
    bouguer_2d[mask_nan] = bouguer_nearest[mask_nan]
    print(f"  -> {mask_nan.sum()} titik NaN diisi dengan nearest.")
else:
    bouguer_2d = bouguer_cubic

# =============================================================================
# 5. HITUNG VGG DENGAN FFT
# =============================================================================
print("3. Menghitung VGG dengan FFT + filter...")

lat_mean = np.mean(lat_grid)
dx = (5/60) * 111320 * np.cos(np.radians(lat_mean))
dy = (5/60) * 111320
ny, nx = bouguer_2d.shape

# a. Low-pass filter sebelum FFT
print(f"   -> Low-pass filter sigma={SIGMA_BOUGUER}...")
bouguer_smooth = gaussian_filter(bouguer_2d, sigma=SIGMA_BOUGUER)

# b. Mean removal
bg_mean     = np.mean(bouguer_smooth)
bg_zeromean = bouguer_smooth - bg_mean

# c. Zero-pad 25% reflect
pad_y = ny // 4
pad_x = nx // 4
bg_padded = np.pad(bg_zeromean, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')

# d. Cosine taper pada domain padded
ny_p, nx_p = bg_padded.shape
taper_x    = tukey(nx_p, alpha=0.2)
taper_y    = tukey(ny_p, alpha=0.2)
taper_2d   = np.outer(taper_y, taper_x)
bg_tapered = bg_padded * taper_2d

# e. Forward FFT
bg_fft = fft2(bg_tapered)

# f. Operator gradien pada domain padded
print(f"   -> Operator gradien k_cut={K_CUT_FACTOR} * K_max...")
kx_p       = 2 * np.pi * fftfreq(nx_p, d=dx)
ky_p       = 2 * np.pi * fftfreq(ny_p, d=dy)
KX_p, KY_p = np.meshgrid(kx_p, ky_p)
K_p        = np.sqrt(KX_p**2 + KY_p**2)
k_max      = np.max(K_p)
k_cut      = K_CUT_FACTOR * k_max
K_filtered = K_p * np.exp(-(K_p**2) / (2 * k_cut**2))
vgg_fft    = bg_fft * K_filtered
vgg_full   = np.real(ifft2(vgg_fft))

# g. Crop balik ke ukuran asli
vgg_2d = vgg_full[pad_y:pad_y+ny, pad_x:pad_x+nx]

# h. Konversi ke Eötvös
vgg_2d_eotvos = vgg_2d * 10000

# i. Clipping
if CLIP_VGG:
    print(f"   -> Clipping VGG ke [{VGG_CLIP_MIN}, {VGG_CLIP_MAX}] Eötvös...")
    vgg_2d_eotvos = np.clip(vgg_2d_eotvos, VGG_CLIP_MIN, VGG_CLIP_MAX)

# =============================================================================
# 6. SIMPAN CSV MASTER
# =============================================================================
df_master_5m = pd.DataFrame({
    'lon':           lon_mesh.flatten(),
    'lat':           lat_mesh.flatten(),
    'Bouguer_mGal':  bouguer_2d.flatten(),
    'VGG_Eotvos':    vgg_2d_eotvos.flatten()
})
df_master_5m.to_csv("BandaArc_Merged_Bouguer_VGG_5Min.csv", index=False)
print("   -> CSV 5-menit disimpan.")

# =============================================================================
# 7. EKSPOR KE GRABLOX2
# =============================================================================
def export_to_grablox(lon_m, lat_m, val_m, filename, title, header_code):
    lon_0, lat_0 = 118.0, -11.0
    lon_flat     = lon_m.flatten()
    lat_flat     = lat_m.flatten()
    val_flat     = val_m.flatten()
    with open(filename, 'w') as f:
        f.write(f"{title}\n")
        f.write(f"{len(val_flat)} 1 2 3 4 {header_code}\n")
        for i in range(len(val_flat)):
            x_km = (lon_flat[i] - lon_0) * 111.32 * math.cos(math.radians(lat_flat[i]))
            y_km = (lat_flat[i] - lat_0) * 111.32
            f.write(f"  {x_km:10.3f}  {y_km:10.3f}  {-0.100:10.3f}  {val_flat[i]:12.6f}\n")

print("4. Menyimpan file Grablox 5-menit...")
export_to_grablox(lon_mesh, lat_mesh, bouguer_2d,
                  'BandaArc_Bouguer_Fix_5Min.DAT', 'BandaArc_Bouguer_5Min', 0)
export_to_grablox(lon_mesh, lat_mesh, vgg_2d_eotvos,
                  'BandaArc_Gradient_Fix_5Min.GAT', 'BandaArc_VGG_5Min', 1)

print("5. Menyimpan file Grablox 10-menit...")
lon_10m    = lon_mesh[::2, ::2]
lat_10m    = lat_mesh[::2, ::2]
bouguer_10m = bouguer_2d[::2, ::2]
vgg_10m    = vgg_2d_eotvos[::2, ::2]
export_to_grablox(lon_10m, lat_10m, bouguer_10m,
                  'BandaArc_Bouguer_Fix_10Min.DAT', 'BandaArc_Bouguer_10Min', 0)
export_to_grablox(lon_10m, lat_10m, vgg_10m,
                  'BandaArc_Gradient_Fix_10Min.GAT', 'BandaArc_VGG_10Min', 1)

# =============================================================================
# 8. VISUALISASI
# =============================================================================
print("6. Menggambar peta...")

bouguer_visual = gaussian_filter(bouguer_2d, sigma=0.5)
vgg_visual     = gaussian_filter(vgg_2d_eotvos, sigma=0.8)
vgg_valid      = vgg_visual[np.isfinite(vgg_visual)]
vgg_abs        = min(max(abs(vgg_valid.min()), abs(vgg_valid.max())), 100)
norm_v         = TwoSlopeNorm(vmin=-vgg_abs, vcenter=0, vmax=vgg_abs)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
extents = [lon_mesh.min(), lon_mesh.max(), lat_mesh.min(), lat_mesh.max()]

# Panel Bouguer
im1 = axes[0].imshow(bouguer_visual, extent=extents, origin='lower',
                     cmap='jet', interpolation='bicubic', aspect='equal')
axes[0].set_title('Anomali Bouguer (mGal)\n[SIO+BATNAS & GGMplus+DEMNAS]',
                  fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Longitude (°E)')
axes[0].set_ylabel('Latitude (°N)')
axes[0].grid(True, linestyle='--', alpha=0.5, color='white')
cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cbar1.set_label('mGal', rotation=270, labelpad=20)
b_valid = bouguer_visual[np.isfinite(bouguer_visual)]
axes[0].text(0.02, 0.03,
             f"min={b_valid.min():.1f}\nmax={b_valid.max():.1f}\nstd={b_valid.std():.1f}",
             transform=axes[0].transAxes, fontsize=8, va='bottom',
             bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

# Panel VGG
im2 = axes[1].imshow(vgg_visual, extent=extents, origin='lower',
                     cmap='RdBu_r', norm=norm_v,
                     interpolation='bicubic', aspect='equal')
axes[1].set_title('Gradien Gaya Berat Vertikal — VGG\n[FFT, sigma=1.0, k_cut=0.3]',
                  fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Longitude (°E)')
axes[1].set_ylabel('Latitude (°N)')
axes[1].grid(True, linestyle='--', alpha=0.4, color='gray')
cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
cbar2.set_label('Eötvös', rotation=270, labelpad=20)
axes[1].text(0.02, 0.03,
             f"min={vgg_valid.min():.1f}\nmax={vgg_valid.max():.1f}\nstd={vgg_valid.std():.1f}",
             transform=axes[1].transAxes, fontsize=8, va='bottom',
             bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

plt.tight_layout()
plt.savefig("BandaArc_Gravity_VGG_Map_Final.png", dpi=300, bbox_inches='tight')
print("\nSelesai! Semua file berhasil dibuat.")
print(f"Bouguer : min={b_valid.min():.2f}, max={b_valid.max():.2f} mGal")
print(f"VGG     : min={vgg_valid.min():.2f}, max={vgg_valid.max():.2f} Eötvös")