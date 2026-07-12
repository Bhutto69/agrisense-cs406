# AgriSense 🌾
**Smart & Precision Agriculture Using Satellite Remote Sensing and Weather-Aware Image Analysis**

CS406 — Digital Image Processing | Spring 2026 | UMT Lahore

---

## Overview
AgriSense is a six-stage satellite image-processing pipeline that monitors crop health over ~500 hectares of mixed wheat, cotton, and rice fields in Sahiwal, Punjab, Pakistan using freely available Sentinel-2 imagery and open-source Python tools.

## Pipeline Architecture

## Results Summary
| Task | Method | Key Metric |
|------|--------|------------|
| T1: Preprocessing | GEE + SCL cloud mask | 51 images (Season 1), 36 (Season 2) |
| T2: Restoration | Dark Channel Prior | PSNR=21.3dB, SSIM=0.853 |
| T3: Indices | NDVI/EVI/NDWI/NDRE/SAVI/LAI | Mean NDVI=0.447 |
| T4: Segmentation | Random Forest | CV Accuracy=1.000 |
| T5: Change Detection | NDVI Diff + CVA | Kappa>0.95 |
| T6: Super-Resolution | SRCNN | PSNR=30.8dB, SSIM=0.836 |

## Study Area
- **Location:** Sahiwal District, Punjab, Pakistan
- **Coordinates:** 73.10°E–73.20°E, 30.65°N–30.75°N
- **Area:** ~500 hectares
- **Crops:** Wheat, cotton, rice

## Installation
```bash
git clone https://github.com/Bhutto69/agrisense-cs406.git
cd agrisense-cs406
python -m venv agrisense_env
agrisense_env\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
# Run each task in order
python task1_preprocess/task1_preprocess.py
python task2_restoration/task2_restoration.py
python task3_indices/task3_indices.py
python task4_segmentation/task4_segmentation.py
python task5_change_detection/task5_change_detection.py
python task6_superres/task6_superres.py
```

## Requirements
- Python 3.10+
- Google Earth Engine account (register at earthengine.google.com)
- See `requirements.txt` for full package list

## Dataset
- **Sentinel-2 L2A** via Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)
- **Study period:** Oct 2022 – Apr 2023 (Season 1), Oct 2023 – Apr 2024 (Season 2)

## Repository Structure

## Submission Links
- **GitHub:** https://github.com/Bhutto69/agrisense-cs406
- **Overleaf Report:** https://www.overleaf.com/read/hmdtjmfrmmbv#7b0a81

## Authors
**Aamir Azeem** (F2022266248) | **Hamza Shahid** (F2022266067)
CS406 Digital Image Processing | Section V3 | UMT Lahore | Spring 2026
Instructor: Dr. Jameel Ahmad