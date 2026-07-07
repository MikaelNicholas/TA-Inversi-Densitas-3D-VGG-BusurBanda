import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def calculate_seawater_density_layer(depth, latitude, layer_depth):
    """
    Calculate seawater density for specific layer menggunakan rumus yang dikoreksi:
    α(φ) = 27.91 − 2.06 exp[−(0.0161|φ|)⁵]
    β(φ) = 0.00637 + 0.00828 exp[−(0.017|φ|)⁴.⁶⁶]
    υ(φ) = 0.964 − 0.091 exp[−(0.016|φ|)⁵]
    μ(φ) = 0.928 − 0.079 × cos(0.053φ)
    
    ρ(D,φ) = 1000 + α(φ) × {μ(φ) + (1−μ(φ)/2) × [1 + tanh(0.00988D − 1.01613)]} + β(φ) × D^υ(φ)
    """
    D = layer_depth
    
    # Step 1: Calculate parameters α, β, υ, μ
    phi_abs = abs(latitude)
    alpha = 27.91 - 2.06 * math.exp(-(0.0161 * phi_abs) ** 5)
    beta = 0.00637 + 0.00828 * math.exp(-(0.017 * phi_abs) ** 4.66)
    upsilon = 0.964 - 0.091 * math.exp(-(0.016 * phi_abs) ** 5)
    mu = 0.928 - 0.079 * math.cos(0.053 * latitude)
    
    # Step 2: Calculate the main components
    tanh_component = math.tanh(0.00988 * D - 1.01613)
    bracket_component = ((1 - mu) / 2) * (1 + tanh_component)
    mu_bracket = mu + bracket_component
    depth_power = beta * (D ** upsilon)
    
    # Step 3: Final density calculation
    density = 1000 + alpha * mu_bracket + depth_power
    return density

def calculate_3d_density_profile(depth, latitude):
    """
    Calculate 3D density profile dengan 84 layer @ 100m
    Returns: average density untuk seluruh water column
    """
    total_depth = depth
    num_layers = min(84, int(total_depth // 100) + 1)  # Maksimal 84 layer
    
    if num_layers == 0:
        return 1030.0  # Default untuk depth sangat dangkal
    
    layer_densities = []
    layer_depths = []
    
    for i in range(num_layers):
        layer_top = i * 100
        layer_bottom = min((i + 1) * 100, total_depth)
        layer_mid = (layer_top + layer_bottom) / 2
        
        layer_density = calculate_seawater_density_layer(depth, latitude, layer_mid)
        layer_densities.append(layer_density)
        layer_depths.append(layer_mid)
    
    avg_density = np.mean(layer_densities)
    return avg_density, layer_densities, layer_depths

# Read bathymetry data
print("Reading bathymetry data...")
df = pd.read_csv('bathymetry_fixed_conversion.txt', sep='\t')
print(f"Total points: {len(df):,}")

# Calculate densities
print("Calculating 3D seawater densities with CORRECTED formula...")

df['density_3d_layers'] = df.apply(
    lambda row: calculate_3d_density_profile(row['depth'], row['latitude'])[0], 
    axis=1
)

df['density_constant'] = 1030.0

# =====================================================================
# THE SORTING FIX: Aligns perfectly with NOAA GlobSed format
# Scans row by row (Latitude first), moving West to East (Longitude)
# =====================================================================
print("\nSorting data to match sediment South-to-North, West-to-East order...")
df = df.sort_values(by=['latitude', 'longitude'], ascending=[True, True])
# =====================================================================

# Validation
print("\n=== 3D DENSITY CALCULATION RESULTS ===")
ocean_df = df[df['depth'] > 0]
print(f"Ocean points: {len(ocean_df):,}")
print(f"Ocean depth range: {ocean_df['depth'].min():.1f} to {ocean_df['depth'].max():.1f} m")
print(f"Ocean 3D density range: {ocean_df['density_3d_layers'].min():.1f} to {ocean_df['density_3d_layers'].max():.1f} kg/m³")

# Export results
output_file = 'bathymetry_3d_density_corrected.txt'
df.to_csv(output_file, sep='\t', index=False, float_format='%.5f')
print(f"\n✅ COMPLETE! File saved: {output_file}")

# 3D Distribution Analysis
print(f"\n🌊 3D DENSITY DISTRIBUTION ANALYSIS:")
if len(ocean_df) > 0:
    density_ranges = [
        (1020, 1025), (1025, 1030), (1030, 1035), (1035, 1040),
        (1040, 1045), (1045, 1050), (1050, 1055), (1055, 1060),
        (1060, 1065)
    ]
    print("Density Range (kg/m³)\tPoints\tPercentage")
    print("-" * 50)
    total_ocean = len(ocean_df)
    for d_min, d_max in density_ranges:
        range_data = ocean_df[
            (ocean_df['density_3d_layers'] >= d_min) & 
            (ocean_df['density_3d_layers'] < d_max)
        ]
        count = len(range_data)
        percentage = (count / total_ocean) * 100
        print(f"{d_min}-{d_max}\t\t{count:,}\t{percentage:.1f}%")

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(ocean_df['density_3d_layers'], bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Density (kg/m³)')
plt.ylabel('Frequency')
plt.title('3D Seawater Density Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.scatter(ocean_df['depth'], ocean_df['density_3d_layers'], alpha=0.5, s=1, color='red')
plt.xlabel('Depth (m)')
plt.ylabel('Density (kg/m³)')
plt.title('Density vs Depth')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.scatter(ocean_df['latitude'], ocean_df['density_3d_layers'], alpha=0.5, s=1, color='green')
plt.xlabel('Latitude (°)')
plt.ylabel('Density (kg/m³)')
plt.title('Density vs Latitude')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.hist(ocean_df['depth'], bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('Depth (m)')
plt.ylabel('Frequency')
plt.title('Ocean Depth Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
sample_point = ocean_df.iloc[0]
_, layer_densities, layer_depths = calculate_3d_density_profile(sample_point['depth'], sample_point['latitude'])
plt.plot(layer_densities, layer_depths, 'o-', linewidth=2, markersize=4)
plt.gca().invert_yaxis()
plt.xlabel('Density (kg/m³)')
plt.ylabel('Depth (m)')
plt.title(f"Profile: {sample_point['latitude']:.2f}°N, {sample_point['depth']:.0f}m")
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
latitudes = np.arange(-11, -2, 0.5)
alphas = [27.91 - 2.06 * math.exp(-(0.0161 * abs(lat)) ** 5) for lat in latitudes]
betas = [0.00637 + 0.00828 * math.exp(-(0.017 * abs(lat)) ** 4.66) for lat in latitudes]
upsilons = [0.964 - 0.091 * math.exp(-(0.016 * abs(lat)) ** 5) for lat in latitudes]
mus = [0.928 - 0.079 * math.cos(0.053 * lat) for lat in latitudes]

plt.plot(latitudes, alphas, 'r-', label='α(φ)')
plt.plot(latitudes, betas, 'g-', label='β(φ)')
plt.plot(latitudes, upsilons, 'b-', label='υ(φ)')
plt.plot(latitudes, mus, 'm-', label='μ(φ)')
plt.xlabel('Latitude (°)')
plt.title('Parameters vs Latitude')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('3d_density_corrected_analysis.png', dpi=300, bbox_inches='tight')
plt.show()