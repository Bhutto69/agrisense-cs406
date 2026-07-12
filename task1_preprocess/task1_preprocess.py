"""
AgriSense - Task 1: Data Acquisition and Radiometric Pre-Processing
CS406 - Digital Image Processing, Spring 2026
"""

import ee
import os

# ─── Initialize GEE ───────────────────────────────────────────────────────────
ee.Initialize(project='agrisense-cs406')
print("GEE initialized successfully")

# ─── Study Area: Sahiwal District, Punjab, Pakistan (~500 hectares) ───────────
study_area = ee.Geometry.Rectangle([73.10, 30.65, 73.20, 30.75])

# ─── Date Ranges (two growing seasons) ────────────────────────────────────────
season1_start = '2022-10-01'
season1_end   = '2023-04-30'
season2_start = '2023-10-01'
season2_end   = '2024-04-30'

# ─── Cloud Masking Function (using SCL band) ──────────────────────────────────
def mask_s2_clouds(image):
    scl = image.select('SCL')
    # SCL classes: 4=vegetation, 5=bare soil, 6=water, 7=unclassified
    # Exclude: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=thin cirrus
    clear_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(clear_mask).divide(10000).copyProperties(image, ['system:time_start'])

# ─── Load Sentinel-2 Collection ───────────────────────────────────────────────
def get_s2_collection(start_date, end_date):
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(study_area)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              .map(mask_s2_clouds))

s2_season1 = get_s2_collection(season1_start, season1_end)
s2_season2 = get_s2_collection(season2_start, season2_end)

print(f"Season 1 images: {s2_season1.size().getInfo()}")
print(f"Season 2 images: {s2_season2.size().getInfo()}")

# ─── Create Median Composites (BOA Reflectance) ───────────────────────────────
bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

composite_s1 = s2_season1.select(bands).median().clip(study_area)
composite_s2 = s2_season2.select(bands).median().clip(study_area)

print("Composites created successfully")

# ─── Export to Google Drive ───────────────────────────────────────────────────
def export_composite(composite, description, folder='AgriSense_Data'):
    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=description,
        folder=folder,
        region=study_area,
        scale=10,
        crs='EPSG:32642',
        maxPixels=1e9
    )
    task.start()
    print(f"Export started: {description} — check Google Drive > {folder}")
    return task

task1 = export_composite(composite_s1, 'S2_Season1_BOA_Composite')
task2 = export_composite(composite_s2, 'S2_Season2_BOA_Composite')

# ─── Quick Stats for Report ───────────────────────────────────────────────────
stats = composite_s1.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=study_area,
    scale=10,
    maxPixels=1e9
)
print("\nSeason 1 Mean BOA Reflectance per band:")
print(stats.getInfo())

print("\nTask 1 complete. Check Google Drive for exported GeoTIFFs.")