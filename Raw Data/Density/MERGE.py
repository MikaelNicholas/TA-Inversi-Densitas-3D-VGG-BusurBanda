import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# --- KONFIGURASI FILE ---
file_batnas = 'Water_3D_Regridded.txt'       # Data Air
file_sediment = 'BandaArc_3D_Sediment_5min.txt'       # Data Sedimen
file_complete = 'CompleteDensity.txt' # Data Kerak/Lithosphere
output_5min_200km = 'BandaArc_5min_200km.txt'
output_10min_200km = 'BandaArc_10min_200km.txt'
output_8layer_125km = 'BandaArc_10min_8layer_GRABLOX.txt'

# Konstanta Geofisika
DENSITY_MANTLE = 3.300  # g/cm3 (untuk kedalaman > kerak)

def load_and_preprocess(file, names):
    df = pd.read_csv(file, sep='\s+', skiprows=1, names=names)
    # Konversi ke g/cm3 jika data dalam kg/m3
    if df['density'].max() > 500:
        df['density'] = df['density'] / 1000.0
    return df

# 1. LOAD SEMUA DATA
# Sesuaikan names dengan header file lu: [lon, lat, depth, density]
names = ['lon', 'lat', 'depth', 'density']
df_air = load_and_preprocess(file_batnas, names)
df_sed = load_and_preprocess(file_sediment, names)
df_com = load_and_preprocess(file_complete, names)

# Gabungkan semua data mentah menjadi satu dataframe besar
df_all = pd.concat([df_air, df_sed, df_com])

def generate_model(res_arcmin, max_depth_km, is_8layer=False):
    # Buat grid koordinat berdasarkan area lu (118-134 BT, -11 ke -2 LS)
    res_deg = res_arcmin / 60.0
    lons = np.arange(118.0, 134.0 + res_deg, res_deg)
    lats = np.arange(-11.0, -2.0 + res_deg, res_deg)
    
    # Tentukan kedalaman sampling
    if is_8layer:
        # Sampling di titik tengah 8 layer (125km / 8 = 15.625km per layer)
        layer_thick = max_depth_km / 8
        target_depths = np.array([layer_thick * (i + 0.5) for i in range(8)])
    else:
        # Sampling standar tiap 1km atau sesuai kebutuhan lu
        target_depths = np.linspace(0, max_depth_km, int(max_depth_km + 1))

    final_data = []

    for lat in lats:
        for lon in lons:
            # Cari data terdekat di grid (karena data mentah mungkin gak pas di grid menit)
            # Filter data pada koordinat tersebut
            mask = (np.isclose(df_all['lon'], lon, atol=0.01)) & \
                   (np.isclose(df_all['lat'], lat, atol=0.01))
            point_data = df_all[mask].sort_values('depth')

            if point_data.empty:
                # Jika koordinat kosong, isi default mantel
                layer_vals = np.full(len(target_depths), DENSITY_MANTLE)
            else:
                depths = point_data['depth'].values
                densities = point_data['density'].values
                
                # Bersihkan duplikat depth agar interp1d tidak error
                depths, indices = np.unique(depths, return_index=True)
                densities = densities[indices]

                # Interpolasi Linier
                # fill_value: atas pakai data air teratas, bawah pakai mantel standar
                f = interp1d(depths, densities, kind='linear', bounds_error=False, 
                             fill_value=(densities[0], DENSITY_MANTLE))
                layer_vals = f(target_depths)

            if is_8layer:
                # GRABLOX minta satu kolom memanjang
                final_data.extend(layer_vals)
            else:
                # Format 3D standar (Lon Lat Depth Density)
                for d, den in zip(target_depths, layer_vals):
                    final_data.append([lon, lat, d, den])

    return np.array(final_data)

# --- PROSES DAN SIMPAN ---

print("Memproses 10 menit 8 layer (125km)...")
data_8l = generate_model(10, 125, is_8layer=True)
np.savetxt(output_8layer_125km, data_8l, fmt='%.4f')

print("Memproses 10 menit standar (200km)...")
data_10m = generate_model(10, 200)
pd.DataFrame(data_10m).to_csv(output_10min_200km, sep='\t', index=False, header=False)

print("Memproses 5 menit standar (200km)... Ini agak lama...")
data_5m = generate_model(5, 200)
pd.DataFrame(data_5m).to_csv(output_5min_200km, sep='\t', index=False, header=False)

print("Semua proses selesai!")