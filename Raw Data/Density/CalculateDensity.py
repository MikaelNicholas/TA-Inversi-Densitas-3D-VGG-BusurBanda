import pandas as pd
import numpy as np

# --- 1. SETTINGS ---
sediment_file = 'BandaArc_Sediment_ThicknessRev.csv'
# If you have a GEBCO bathymetry CSV (lon, lat, depth_m), load it here later.
# bathymetry_file = 'BandaArc_Bathymetry.csv' 
output_txt = 'BandaArc_3D_Sediment_5min.txt'

# --- 2. DENSITY FUNCTION (Tenzer & Gladkikh, 2014) ---
def calc_sediment_density(ds_m, D_m):
    """
    ds_m: Depth below the seafloor in meters (Sediment depth)
    D_m: Ocean depth in meters (Bathymetry)
    Returns: Density in kg/m3
    """
    # Equation 6
    density = (1.66 - 5.1e-5 * D_m + 0.0037 * (ds_m ** 0.766)) * 1000
    
    # Bedrock contrast correction: Cap density at 2750 kg/m3 for deep sediments
    if density > 2750.0:
        return 2750.0
    return density

# --- 3. LOAD DATA ---
print(f"Loading sediment data from {sediment_file}...")
df_sed = pd.read_csv(sediment_file)

# Temporary mock for Bathymetry (D) until you process GEBCO
# Assuming an average Banda Arc ocean depth of 3000 meters for testing
df_sed['ocean_depth_m'] = 3000.0 

# --- 4. EXTRUDE 3D VOLUME ---
print(f"Calculating 'per km' 3D density volume...")

with open(output_txt, 'w') as f:
    # Header aligned for VGG modeling
    f.write(f"{'longitude':<12} {'latitude':<12} {'depth_bsf_km':<15} {'density_kgm3':<15}\n")
    
    # Counter for tracking progress
    nodes_processed = 0
    total_layers = 0
    
    for row in df_sed.itertuples():
        lon = row.lon
        lat = row.lat
        total_thickness_m = row.thickness
        D_m = row.ocean_depth_m
        
        # Skip locations with no sediment (like onshore areas or exposed rock)
        if pd.isna(total_thickness_m) or total_thickness_m <= 0:
            continue
            
        nodes_processed += 1
        
        # Create vertical layers "per km" (every 1000 meters)
        # We start at 0 (seafloor) and go down to the total thickness
        # np.arange creates steps: 0, 1000, 2000, etc.
        vertical_steps_m = np.arange(0, total_thickness_m + 1000, 1000)
        
        for ds_m in vertical_steps_m:
            # Ensure we don't calculate past the actual bottom of the sediment
            if ds_m > total_thickness_m:
                ds_m = total_thickness_m
                
            rho = calc_sediment_density(ds_m, D_m)
            
            # Convert ds_m back to km for the output text file
            depth_km = ds_m / 1000.0
            
            f.write(f"{lon:<12.5f} {lat:<12.5f} {depth_km:<15.3f} {rho:<15.2f}\n")
            total_layers += 1

print("-" * 30)
print(f"SUCCESS! 3D Sediment file generated: {output_txt}")
print(f"Horizontal nodes processed: {nodes_processed}")
print(f"Total 3D voxels (1km vertical steps) created: {total_layers}")
print("-" * 30)