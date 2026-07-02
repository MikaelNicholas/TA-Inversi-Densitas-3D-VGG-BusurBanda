# Data dan Kode Tugas Akhir - Pemodelan 3D VGG dan Subsiden Tektonik Busur Banda

Repositori ini memuat kode pengolahan data dan model hasil inversi 3D untuk Tugas Akhir "Pemodelan 3D Gradien Gayaberat Vertikal dan Subsiden Tektonik pada Zona Konvergensi Tiga Lempeng Tektonik di Indonesia".

**Penulis:** Mikael Nicholas Prakoso (NRP 5016221042)
**Departemen Teknik Geomatika, Institut Teknologi Sepuluh Nopember (ITS)**
**Dosen Pembimbing:** Ira Mutiara Anjasmara, S.T., M.Phil, Ph.D.

## Kode GMT
- `LokasiPenelitian.bat` - Script GMT untuk membuat peta lokasi penelitian wilayah Busur Banda.
- `earth_relief_10m.grd` - Grid topografi/batimetri resolusi 10 menit dari GMT, digunakan sebagai basemap pada peta lokasi penelitian.

## Kode Python
- `FFT.py` - Pipeline pengolahan Vertical Gravity Gradient (VGG) dari data anomali Bouguer menggunakan metode Fast Fourier Transform (FFT), meliputi reflect padding, mean removal, tapering, operator diferensiasi spektral dengan Gaussian damping, IFFT, cropping, konversi satuan ke Eotvos, dan clipping.
- `Visualisasi.py` - Script visualisasi peta anomali Bouguer, peta VGG, peta densitas per layer, scatter plot perbandingan data observasi dan komputasi, serta penampang (cross-section) hasil inversi.

## Data Hasil Inversi, Bouguer, dan Vertical Gravity Gradient (VGG) (GRABlox2)
- `Grablox2_current_iter.blx` - File model blok 3D hasil parameterisasi awal (grid density model), berisi koordinat pusat tiap blok (x, y, z), dimensi blok, dan nilai densitas per blok sebelum/selama proses inversi. Format kolom: x, y, z-height, x-center, y-center, z-thickness, index, density.
- `Grablox2_current_iter.dat` - Data anomali gravitasi (Bouguer) hasil GRABLOX 2.1, berisi koordinat titik amat (x, y, h) dan tiga nilai gravitasi: g comp (gravity computed/hasil forward modeling), g base (base level/regional), dan g meas (gravity measured/data observasi).
- `Grablox2_current_iter.gat` - Data Vertical Gravity Gradient (VGG) hasil GRABLOX 2.12, berisi koordinat titik amat (x, y, h) dan nilai gzz comp (VGG hasil komputasi model) serta gzz meas (VGG hasil observasi/turunan dari data Bouguer).
- `Grablox2_current_iter.inp` - File parameter input untuk GRABLOX2, mendefinisikan geometri model (posisi, dimensi, jumlah divisi blok 103x61x8), batas nilai densitas (1.0-3.4 g/cm3), parameter inversi (Occam density optimization), dan referensi ke file data Bouguer & gradien yang digunakan.
- `Grablox2_current_iter.out` - File log/hasil akhir inversi, berisi ringkasan informasi model (dimensi, diskretisasi, densitas rata-rata per layer), parameter inversi yang dipakai, serta metrik akurasi (RMS error data, RMS error gradien, RMS error model) dan waktu komputasi.

## Data Tambahan
- `Banda_Arc_Slab2_Merged.grd` - Grid geometri slab (kedalaman megathrust) dari model Slab2 untuk zona Busur Banda, digunakan sebagai data pembanding/validasi visual terhadap arah dan pola subsiden tektonik hasil inversi. Grid ini dioverlaykan pada peta densitas dan penampang (cross-section) untuk mengecek kesesuaian arah subsiden dengan geometri slab.
