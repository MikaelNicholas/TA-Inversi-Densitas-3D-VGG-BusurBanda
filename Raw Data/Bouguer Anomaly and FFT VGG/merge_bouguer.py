"""
merge_bouguer.py
================
Menggabungkan data Bouguer anomaly laut (marine) dan darat (land),
mengecek duplikat koordinat, dan menghasilkan file gabungan yang
sudah diurutkan berdasarkan koordinat.

Area penelitian: lon 118–134, lat -11 s.d. -2
Resolusi grid  : 5 menit busur → target 21.037 titik
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
marine = pd.read_csv("marine_bouguer_anomaly_result.csv")
land   = pd.read_csv("gravity_anomaly_result.csv")

print("=" * 55)
print("LAPORAN MERGE BOUGUER ANOMALY")
print("=" * 55)
print(f"\n[INFO] Marine rows : {len(marine):>7,}")
print(f"[INFO] Land rows   : {len(land):>7,}")
print(f"[INFO] Total raw   : {len(marine) + len(land):>7,}")

# ─────────────────────────────────────────────
# 2. STANDARISASI KOLOM → pakai (lon, lat, DeltaGB_mGal)
# ─────────────────────────────────────────────
marine_std = marine.rename(columns={"lon": "lon", "lat": "lat"})[["lon", "lat", "DeltaGB_mGal"]].copy()
marine_std["source"] = "marine"

land_std = land.rename(columns={"Longitude": "lon", "Latitude": "lat"})[["lon", "lat", "DeltaGB_mGal"]].copy()
land_std["source"] = "land"

# ─────────────────────────────────────────────
# 3. GABUNGKAN
# ─────────────────────────────────────────────
combined = pd.concat([marine_std, land_std], ignore_index=True)

# ─────────────────────────────────────────────
# 4. DETEKSI DUPLIKAT KOORDINAT
# ─────────────────────────────────────────────
# Bulatkan ke 6 desimal untuk menghindari floating-point mismatch kecil
combined["lon_r"] = combined["lon"].round(6)
combined["lat_r"] = combined["lat"].round(6)

dup_mask = combined.duplicated(subset=["lon_r", "lat_r"], keep=False)
duplicates = combined[dup_mask].sort_values(["lon_r", "lat_r"])

print(f"\n[CEK DUPLIKAT]")
print(f"  Titik dengan koordinat duplikat  : {len(duplicates):,} baris")
print(f"  Jumlah pasangan duplikat unik    : {len(duplicates) // 2:,} titik")

if len(duplicates) > 0:
    print("\n  Contoh 10 duplikat pertama:")
    print(duplicates.head(10).to_string(index=False))

    # Simpan laporan duplikat
    duplicates.to_csv("bouguer_duplicates_report.csv", index=False)
    print("\n  [SIMPAN] Laporan duplikat → bouguer_duplicates_report.csv")

# ─────────────────────────────────────────────
# 5. HAPUS DUPLIKAT
#    Prioritas: marine dipertahankan (keep='first' karena marine concat duluan)
# ─────────────────────────────────────────────
combined_clean = combined.drop_duplicates(subset=["lon_r", "lat_r"], keep="first").copy()
combined_clean = combined_clean.drop(columns=["lon_r", "lat_r"])

print(f"\n[SETELAH DEDUPLIKASI]")
print(f"  Sisa baris       : {len(combined_clean):,}")
print(f"  Target (21.037)  : 21,037")
diff = len(combined_clean) - 21037
print(f"  Selisih vs target: {diff:+,}")

# ─────────────────────────────────────────────
# 6. FILTER AREA PENELITIAN (118–134, -11 s.d. -2)
# ─────────────────────────────────────────────
mask_area = (
    (combined_clean["lon"] >= 118.0) & (combined_clean["lon"] <= 134.0) &
    (combined_clean["lat"] >= -11.0) & (combined_clean["lat"] <= -2.0)
)
combined_area = combined_clean[mask_area].copy()

print(f"\n[FILTER AREA 118–134, -11 s.d. -2]")
print(f"  Titik dalam area : {len(combined_area):,}")

# ─────────────────────────────────────────────
# 7. URUTKAN: lat ascending (utara → selatan), lon ascending
# ─────────────────────────────────────────────
combined_area = combined_area.sort_values(["lat", "lon"], ascending=[True, True]).reset_index(drop=True)

# ─────────────────────────────────────────────
# 8. SIMPAN OUTPUT UTAMA
# ─────────────────────────────────────────────
output_file = "bouguer_anomaly_merged.csv"
combined_area.to_csv(output_file, index=False)

print(f"\n[SIMPAN] Output utama → {output_file}")
print(f"\n  Kolom output: {list(combined_area.columns)}")
print(f"  Baris output: {len(combined_area):,}")
print(f"\n  Statistik DeltaGB_mGal:")
print(combined_area["DeltaGB_mGal"].describe().to_string())

# ─────────────────────────────────────────────
# 9. RINGKASAN KOMPOSISI (marine vs land)
# ─────────────────────────────────────────────
print(f"\n[KOMPOSISI SUMBER DATA]")
print(combined_area["source"].value_counts().to_string())

print("\n" + "=" * 55)
print("SELESAI")
print("=" * 55)