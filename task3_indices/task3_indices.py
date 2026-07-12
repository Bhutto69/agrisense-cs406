"""
AgriSense - Task 3: Vegetation Index Computation and Spectral Feature Engineering
CS406 - Digital Image Processing, Spring 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import ee
import warnings
warnings.filterwarnings('ignore')

# ─── Load Season 1 Composite ──────────────────────────────────────────────────
tif_path = 'data/raw/S2_Season1_BOA_Composite.tif'

with rasterio.open(tif_path) as src:
    data = src.read().astype(np.float32)

# Band order: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
B2   = data[0]   # Blue
B3   = data[1]   # Green
B4   = data[2]   # Red
B5   = data[3]   # RedEdge1
B6   = data[4]   # RedEdge2
B7   = data[5]   # RedEdge3
B8   = data[6]   # NIR
B8A  = data[7]   # NIR narrow
B11  = data[8]   # SWIR1
B12  = data[9]   # SWIR2

# Replace zeros with nan to avoid division errors
for band in [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]:
    band[band == 0] = np.nan

print("Bands loaded successfully")

# ─── Vegetation Index Functions ───────────────────────────────────────────────
def safe_divide(a, b):
    return np.where(b != 0, a / b, np.nan)

# NDVI - Normalized Difference Vegetation Index
NDVI = safe_divide(B8 - B4, B8 + B4)

# EVI - Enhanced Vegetation Index
EVI = 2.5 * safe_divide(B8 - B4, B8 + 6*B4 - 7.5*B2 + 1)

# NDWI - Normalized Difference Water Index
NDWI = safe_divide(B3 - B8, B3 + B8)

# NDRE - Normalized Difference Red Edge
NDRE = safe_divide(B5 - B4, B5 + B4)

# SAVI - Soil Adjusted Vegetation Index
SAVI = 1.5 * safe_divide(B8 - B4, B8 + B4 + 0.5)

# MSAVI - Modified SAVI
MSAVI = (2*B8 + 1 - np.sqrt((2*B8 + 1)**2 - 8*(B8 - B4))) / 2

# LAI - Leaf Area Index (empirical NDVI inversion)
LAI = np.where(NDVI > 0, 3.618 * NDVI - 0.118, np.nan)

print("All indices computed")

# ─── Clip to valid ranges ─────────────────────────────────────────────────────
NDVI  = np.clip(NDVI,  -1, 1)
EVI   = np.clip(EVI,   -1, 1)
NDWI  = np.clip(NDWI,  -1, 1)
NDRE  = np.clip(NDRE,  -1, 1)
SAVI  = np.clip(SAVI,  -1, 1)
MSAVI = np.clip(MSAVI, -1, 1)
LAI   = np.clip(LAI,    0, 8)

# ─── Visualization: All Indices ───────────────────────────────────────────────
indices = {
    'NDVI': (NDVI, 'RdYlGn', 'Vegetation Vigour'),
    'EVI':  (EVI,  'RdYlGn', 'Canopy Structure'),
    'NDWI': (NDWI, 'RdBu',   'Water Content'),
    'NDRE': (NDRE, 'RdYlGn', 'Chlorophyll Status'),
    'SAVI': (SAVI, 'RdYlGn', 'Sparse Vegetation'),
    'LAI':  (LAI,  'YlGn',   'Leaf Area Index'),
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (name, (index, cmap, desc)) in enumerate(indices.items()):
    im = axes[i].imshow(index, cmap=cmap, vmin=-0.5 if name != 'LAI' else 0,
                         vmax=1 if name != 'LAI' else 6)
    axes[i].set_title(f'{name} — {desc}\nMean: {np.nanmean(index):.3f}', fontsize=10)
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.suptitle('Task 3: Vegetation Indices — Sahiwal, Punjab, Pakistan\nAgriSense CS406 Spring 2026', fontsize=13)
plt.tight_layout()
plt.savefig('data/outputs/task3_indices.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to data/outputs/task3_indices.png")

# ─── Stats Table for Report ───────────────────────────────────────────────────
print("\n=== Vegetation Index Statistics ===")
print(f"{'Index':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 45)
for name, (index, _, _) in indices.items():
    print(f"{name:<8} {np.nanmean(index):>8.4f} {np.nanstd(index):>8.4f} "
          f"{np.nanmin(index):>8.4f} {np.nanmax(index):>8.4f}")

# ─── Temporal Profile via GEE ─────────────────────────────────────────────────
print("\nFetching temporal NDVI profile from GEE...")
ee.Initialize(project='agrisense-cs406')

study_area = ee.Geometry.Rectangle([73.10, 30.65, 73.20, 30.75])

def mask_and_ndvi(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    nir = image.select('B8').divide(10000)
    red = image.select('B4').divide(10000)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    return ndvi.updateMask(mask).copyProperties(image, ['system:time_start'])

collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(study_area)
              .filterDate('2022-10-01', '2023-04-30')
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              .map(mask_and_ndvi))

def get_mean_ndvi(image):
    mean = image.reduceRegion(reducer=ee.Reducer.mean(),
                               geometry=study_area, scale=10, maxPixels=1e9)
    return ee.Feature(None, {'ndvi': mean.get('NDVI'),
                              'date': image.date().format('YYYY-MM-dd')})

ndvi_ts = collection.map(get_mean_ndvi).getInfo()
dates = [f['properties']['date'] for f in ndvi_ts['features']]
ndvi_vals = [f['properties']['ndvi'] for f in ndvi_ts['features']]

# ─── Plot Temporal NDVI Profile ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(len(dates)), ndvi_vals, 'g-o', markersize=4, linewidth=1.5)
ax.set_xticks(range(0, len(dates), max(1, len(dates)//10)))
ax.set_xticklabels([dates[i] for i in range(0, len(dates), max(1, len(dates)//10))],
                    rotation=45, ha='right')
ax.set_ylabel('Mean NDVI')
ax.set_title('Temporal NDVI Profile — Sahiwal Growing Season 1 (Oct 2022 – Apr 2023)\nAnnotated Phenological Stages')
ax.grid(alpha=0.3)

# Annotate phenological stages
ax.axvspan(0, 5,   alpha=0.1, color='brown',  label='Sowing')
ax.axvspan(5, 15,  alpha=0.1, color='yellow', label='Tillering')
ax.axvspan(15, 30, alpha=0.1, color='green',  label='Heading/Grain Fill')
ax.axvspan(30, len(dates), alpha=0.1, color='orange', label='Maturity')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('data/outputs/task3_ndvi_temporal.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to data/outputs/task3_ndvi_temporal.png")
print("\nTask 3 complete.")