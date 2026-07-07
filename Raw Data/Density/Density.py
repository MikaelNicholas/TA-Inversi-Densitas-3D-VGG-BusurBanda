import pandas as pd

# --- 1. SETTINGS ---
input_file = 'BandaArc_3D_Volume_5min.txt'
output_file = 'BandaArc_3D_Brocher_Final.txt'

print(f"Reading upsampled data from {input_file}...")

# --- 2. LOAD DATA ---
# Use sep='\s+' for multi-space separation (replaces delim_whitespace)
df = pd.read_csv(input_file, sep='\s+', engine='python')

# --- 3. APPLY BROCHER EQUATIONS ---
print("Calculating Vp (Eq 9) and Density (Eq 1)...")

# Equation 9: Vp from Vs (0 to 4.5 km/s)
def calculate_vp_eq9(vs):
    return (0.9409 + 
            (2.0947 * vs) - 
            (0.8206 * (vs**2)) + 
            (0.2683 * (vs**3)) - 
            (0.0251 * (vs**4)))

# Equation 1: Density from Vp (Nafe-Drake)
def calculate_rho(vp):
    # Returns g/cm3
    rho_gcm3 = (1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5)
    return rho_gcm3 * 1000 # Convert to kg/m3

# Calculating based on your Vs_raw column
df['Vp_Eq9'] = df['Vs_raw'].apply(calculate_vp_eq9)
df['density_kgm3'] = df['Vp_Eq9'].apply(calculate_rho)

# --- 4. SAVE TO TEXT FILE ---
print(f"Writing to {output_file}...")

with open(output_file, 'w') as f:
    # Header with specific widths for alignment
    f.write(f"{'longitude':<15} {'latitude':<15} {'depth_km':<12} {'Vs_raw':<12} {'Vp_Eq9':<12} {'density_kgm3':<15}\n")
    
    # Efficient row-by-row write
    for row in df.itertuples():
        f.write(f"{row.longitude:<15.5f} {row.latitude:<15.5f} {row.depth_km:<12.2f} "
                f"{row.Vs_raw:<12.5f} {row.Vp_Eq9:<12.5f} {row.density_kgm3:<15.2f}\n")

print("-" * 30)
print(f"DONE! Processed {len(df)} nodes.")
print(f"Final output: {output_file}")
print("-" * 30)