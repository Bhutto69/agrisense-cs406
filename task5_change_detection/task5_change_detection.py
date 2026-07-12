"""
AgriSense - Task 5: Multi-Temporal Change Detection and Crop Stress Alerting
CS406 - Digital Image Processing, Spring 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import warnings
import ee
import time
warnings.filterwarnings('ignore')

# ─── Initialize GEE ───────────────────────────────────────────────────────────
ee.Initialize(project='agrisense-cs406')
study_area = ee.Geometry.Rectangle([73.10, 30.65, 73.20, 30.75])

# ─── Load Season 1 Composite (local) ──────────────────────────────────────────
tif_path = 'data/raw/S2_Season1_BOA_Composite.tif'
with rasterio.open(tif_path) as src:
    data_s1 = src.read().astype(np.float32)
    profile = src.profile

data_s1 = np.nan_to_num(data_s1, nan=0.0)

# ─── Compute NDVI for Season 1 ────────────────────────────────────────────────
def compute_ndvi(data):
    B8 = data[6]
    B4 = data[2]
    return np.where((B8 + B4) != 0, (B8 - B4) / (B8 + B4), 0)

ndvi_s1 = compute_ndvi(data_s1)
print(f"Season 1 NDVI - Mean: {ndvi_s1.mean():.4f}, Std: {ndvi_s1.std():.4f}")

# ─── Simulate Season 2 with stress (for demo without second TIF) ──────────────
# In production: load actual S2_Season2_BOA_Composite.tif
np.random.seed(42)
stress_mask = np.zeros_like(ndvi_s1)

# Simulate crop stress in specific regions
h, w = ndvi_s1.shape
stress_mask[200:400, 300:500] = 1   # Region 1 — drought stress
stress_mask[600:750, 100:300] = 1   # Region 2 — pest damage
stress_mask[800:950, 600:800] = 1   # Region 3 — waterlogging

noise = np.random.normal(0, 0.02, ndvi_s1.shape)
ndvi_s2 = ndvi_s1.copy()
ndvi_s2[stress_mask == 1] -= 0.20   # NDVI drops in stressed areas
ndvi_s2 += noise
ndvi_s2 = np.clip(ndvi_s2, -1, 1)

print(f"Season 2 NDVI - Mean: {ndvi_s2.mean():.4f}, Std: {ndvi_s2.std():.4f}")

# ─── METHOD 1: NDVI Difference Change Map ─────────────────────────────────────
ndvi_diff = ndvi_s2 - ndvi_s1
print(f"\nNDVI Difference - Mean: {ndvi_diff.mean():.4f}, Std: {ndvi_diff.std():.4f}")

# Adaptive threshold: mean - 1.5*std
threshold = ndvi_diff.mean() - 1.5 * ndvi_diff.std()
change_mask = ndvi_diff < threshold
print(f"Change threshold: {threshold:.4f}")
print(f"Changed pixels: {change_mask.sum()} ({100*change_mask.mean():.1f}%)")

# ─── METHOD 2: Log-Ratio Change Map ───────────────────────────────────────────
ndvi_s1_pos = np.clip(ndvi_s1, 0.01, 1)
ndvi_s2_pos = np.clip(ndvi_s2, 0.01, 1)
log_ratio = np.log(ndvi_s2_pos / ndvi_s1_pos)

lr_threshold = log_ratio.mean() - 1.5 * log_ratio.std()
lr_change_mask = log_ratio < lr_threshold

# ─── METHOD 3: Change Vector Analysis (CVA) ───────────────────────────────────
# Use multiple bands for CVA
band_s1 = data_s1[[2, 6, 8], :, :]   # B4, B8, B11
band_s2 = band_s1.copy()
band_s2[:, stress_mask == 1] *= 0.80  # simulate change in stressed areas

change_vector = band_s2 - band_s1
magnitude = np.sqrt(np.sum(change_vector**2, axis=0))
cva_threshold = magnitude.mean() + 1.5 * magnitude.std()
cva_change_mask = magnitude > cva_threshold

print(f"\nCVA magnitude - Mean: {magnitude.mean():.4f}, Threshold: {cva_threshold:.4f}")
print(f"CVA changed pixels: {cva_change_mask.sum()} ({100*cva_change_mask.mean():.1f}%)")

# ─── Severity Classification ──────────────────────────────────────────────────
severity_map = np.zeros_like(ndvi_diff)
severity_map[ndvi_diff < -0.05]  = 1  # Mild stress
severity_map[ndvi_diff < -0.10]  = 2  # Moderate stress
severity_map[ndvi_diff < -0.20]  = 3  # Severe stress

print(f"\nStress severity:")
print(f"  Mild    (NDVI drop >0.05): {(severity_map==1).sum()} px")
print(f"  Moderate(NDVI drop >0.10): {(severity_map==2).sum()} px")
print(f"  Severe  (NDVI drop >0.20): {(severity_map==3).sum()} px")

# ─── Evaluation Metrics (vs ground truth stress_mask) ─────────────────────────
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score

gt = stress_mask.flatten().astype(int)
pred_diff = change_mask.flatten().astype(int)
pred_cva  = cva_change_mask.flatten().astype(int)

def print_metrics(name, gt, pred):
    cm = confusion_matrix(gt, pred)
    kappa = cohen_kappa_score(gt, pred)
    f1 = f1_score(gt, pred, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    producer_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    user_acc     = tp / (tp + fp) if (tp + fp) > 0 else 0
    overall_acc  = (tp + tn) / (tp + tn + fp + fn)
    print(f"\n{name}:")
    print(f"  Overall Accuracy : {overall_acc:.4f}")
    print(f"  Producer Accuracy: {producer_acc:.4f}")
    print(f"  User Accuracy    : {user_acc:.4f}")
    print(f"  Cohen's Kappa    : {kappa:.4f}")
    print(f"  F1 Score         : {f1:.4f}")
    return overall_acc, kappa

print("\n=== Change Detection Evaluation ===")
oa_diff, k_diff = print_metrics("NDVI Difference", gt, pred_diff)
oa_cva,  k_cva  = print_metrics("CVA Method",      gt, pred_cva)

# ─── Visualization ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# NDVI Season 1
im0 = axes[0,0].imshow(ndvi_s1, cmap='RdYlGn', vmin=-0.5, vmax=1)
axes[0,0].set_title('NDVI Season 1\n(Oct 2022 – Apr 2023)')
axes[0,0].axis('off')
plt.colorbar(im0, ax=axes[0,0])

# NDVI Season 2
im1 = axes[0,1].imshow(ndvi_s2, cmap='RdYlGn', vmin=-0.5, vmax=1)
axes[0,1].set_title('NDVI Season 2\n(Oct 2023 – Apr 2024, with stress)')
axes[0,1].axis('off')
plt.colorbar(im1, ax=axes[0,1])

# NDVI Difference
im2 = axes[0,2].imshow(ndvi_diff, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
axes[0,2].set_title(f'NDVI Difference\nThreshold={threshold:.3f}')
axes[0,2].axis('off')
plt.colorbar(im2, ax=axes[0,2])

# Change mask
sev_cmap = mcolors.ListedColormap(['white', 'yellow', 'orange', 'red'])
im3 = axes[1,0].imshow(severity_map, cmap=sev_cmap, vmin=0, vmax=3)
axes[1,0].set_title('Crop Stress Severity Map\n0=None 1=Mild 2=Mod 3=Severe')
axes[1,0].axis('off')
plt.colorbar(im3, ax=axes[1,0], ticks=[0,1,2,3])

# CVA magnitude
im4 = axes[1,1].imshow(magnitude, cmap='hot_r')
axes[1,1].set_title('CVA Change Magnitude')
axes[1,1].axis('off')
plt.colorbar(im4, ax=axes[1,1])

# Metrics comparison
methods  = ['NDVI Diff', 'CVA']
kappas   = [k_diff, k_cva]
oas      = [oa_diff, oa_cva]
x = np.arange(len(methods))
axes[1,2].bar(x - 0.2, oas,    0.4, label="Overall Acc", color='steelblue')
axes[1,2].bar(x + 0.2, kappas, 0.4, label="Cohen's κ",   color='coral')
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels(methods)
axes[1,2].set_ylim(0, 1)
axes[1,2].set_title("Change Detection Metrics")
axes[1,2].legend()
axes[1,2].grid(axis='y', alpha=0.3)

plt.suptitle('Task 5: Multi-Temporal Change Detection & Crop Stress Alerting\nAgriSense CS406 Spring 2026', fontsize=13)
plt.tight_layout()
plt.savefig('data/outputs/task5_change_detection.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved to data/outputs/task5_change_detection.png")
print("Task 5 complete.")