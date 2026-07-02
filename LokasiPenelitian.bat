@echo off
echo ==========================================
echo MEMBUAT PETA LOKASI PENELITIAN (ALIGNED)
echo ==========================================

if not exist earth_relief_10m.grd (
    echo Peringatan: File grid tidak ditemukan. Pakai warna solid.
)

echo Memulai plotting...
echo.

gmt begin peta_final png

    :: 1. Basemap + Judul — region diperlebar ke bawah sampai -13.5
    gmt basemap -R116/136/-13.5/0 -JM28c -BWSne+t"PETA LOKASI PENELITIAN" -Bxa2f1 -Bya2f1

    :: 2. Topografi / Warna Dasar
    if exist earth_relief_10m.grd (
        gmt grdimage earth_relief_10m.grd -I+a45+nt0.8 -Cgeo
        gmt grdcontour earth_relief_10m.grd -C1000 -A2000 -W0.5p,white
    ) else (
        gmt coast -Gdarkseagreen -Slightblue -Df
    )

    :: 3. Garis Pantai & Grid
    gmt coast -W1/0.8p,black -N1/1.2p,black -Df
    gmt basemap -Bxa2g2 -Bya2g2

    :: =============================================
    :: GARIS AREA PENELITIAN (tetap di -11.5 s.d -1.7)
    :: =============================================
    echo 117.7 -11.5 > area.txt
    echo 134.3 -11.5 >> area.txt
    echo 134.3 -1.7 >> area.txt
    echo 117.7 -1.7 >> area.txt
    echo 117.7 -11.5 >> area.txt
    gmt plot area.txt -W4p,red
    del area.txt

    :: =============================================
    :: TITIK PENELITIAN (interval 10 menit)
    :: =============================================
    gmt grdmath -R118/134/-11/-2 -I10m 1 = grid_titik.grd
    gmt grd2xyz grid_titik.grd | gmt plot -Sc0.08c -Gred -W0.5p,darkred
    del grid_titik.grd

    :: =============================================
    :: LABEL NAMA LAUT & KOTA
    :: =============================================
    echo 128 -5.5 Banda Sea     | gmt text -F+jCM+f11p,Helvetica-Bold,white -Gblack@30 -C2p -W1p,black
    echo 121 -7.5 Flores Sea    | gmt text -F+jCM+f11p,Helvetica-Bold,white -Gblack@30 -C2p -W1p,black
    echo 126 -1   Molucca Sea   | gmt text -F+jCM+f11p,Helvetica-Bold,white -Gblack@30 -C2p -W1p,black
    echo 130 -2.5 Ceram Sea     | gmt text -F+jCM+f11p,Helvetica-Bold,white -Gblack@30 -C2p -W1p,black
    echo 133 -9   Arafura Sea   | gmt text -F+jCM+f11p,Helvetica-Bold,white -Gblack@30 -C2p -W1p,black
    echo 119.4 -5.1 Makassar    | gmt text -F+jLM+f10p,Helvetica-Bold,white -Gblack@30 -C1.5p -W1p,black
    echo 128.2 -3.7 Ambon       | gmt text -F+jLM+f10p,Helvetica-Bold,white -Gblack@30 -C1.5p -W1p,black
    echo 125.6 -8.6 Dili        | gmt text -F+jLM+f10p,Helvetica-Bold,white -Gblack@30 -C1.5p -W1p,black
    echo 123.6 -10.2 Kupang     | gmt text -F+jLM+f10p,Helvetica-Bold,white -Gblack@30 -C1.5p -W1p,black

    :: =============================================
    :: SCALE BAR — dinaikkan ke -12.6 (lebih dekat ke area)
    :: =============================================
    gmt basemap -Lg128/-12.6+c-11+w200k+u+f+lKM

    :: =============================================
    :: KOMPAS — dinaikkan ke -12.6, bersebelahan dengan scale bar
    :: =============================================
    gmt basemap -Tdg133/-12.6+w1.5c+f2+lB,T,S,U

gmt end

echo.
echo SELESAI!
echo Peta tersimpan di: peta_final.png
pause