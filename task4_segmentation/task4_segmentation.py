"""
AgriSense - Task 4: Image Segmentation and Land-Cover Classification
CS406 - Digital Image Processing, Spring 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import warnings
warnings.filterwarnings('ignore')

from skimage.filters import threshold_otsu
from skimage.morphology import dilation, erosion, opening, disk
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import time

# ─── Load Data ────────────────────────────────────────────────────────────────
tif_path = 'data/raw/S2_Season1_BOA_Composite.tif'

with rasterio.open(tif_path) as src:
    data = src.read().astype(np.float32)

B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 = [data[i] for i in range(10)]

# Replace 0 and nan
data = np.nan_to_num(data, nan=0.0)
B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 = [data[i] for i in range(10)]

print(f"Image shape: {data.shape}")

# ─── Compute NDVI ─────────────────────────────────────────────────────────────
def safe_divide(a, b):
    return np.where((a + b) != 0, (a - b) / (a + b), 0)

NDVI = safe_divide(B8, B4)
EVI  = np.clip(2.5 * np.where((B8 + 6*B4 - 7.5*B2 + 1) != 0,
               (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1), 0), -1, 1)
NDWI = safe_divide(B3, B8)

# ─── METHOD 1: Classical — Otsu + K-means ─────────────────────────────────────
print("\n--- Method 1: Classical Segmentation ---")

# Otsu on NDVI
ndvi_norm = ((NDVI + 1) / 2 * 255).astype(np.uint8)
thresh = threshold_otsu(ndvi_norm)
otsu_mask = ndvi_norm > thresh

# Morphological refinement
otsu_refined = opening(otsu_mask, disk(3))
otsu_refined = dilation(otsu_refined, disk(2))

# Simple rule-based classification from indices
def rule_based_classify(ndvi, ndwi, b11):
    classified = np.zeros(ndvi.shape, dtype=np.uint8)
    classified[ndvi > 0.4]                          = 1  # Dense vegetation
    classified[(ndvi > 0.2) & (ndvi <= 0.4)]        = 2  # Sparse vegetation
    classified[(ndwi > 0.0)]                         = 3  # Water body
    classified[(ndvi < 0.1) & (b11 > 0.2)]          = 4  # Bare soil
    classified[(ndvi < 0.1) & (b11 <= 0.2)]         = 5  # Built-up
    classified[ndvi == 0]                            = 0  # Unclassified
    return classified

classical_map = rule_based_classify(NDVI, NDWI, B11)
print("Classical segmentation done")

# ─── METHOD 2: Random Forest ──────────────────────────────────────────────────
print("\n--- Method 2: Random Forest Classification ---")

# Build feature matrix
h, w = B4.shape
features = np.stack([B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,NDVI,EVI,NDWI], axis=-1)
features_flat = features.reshape(-1, 13)

# Generate pseudo-labels from rule-based map for training
labels_flat = classical_map.flatten()

# Sample balanced subset for training (avoid memory issues)
np.random.seed(42)
valid_mask = labels_flat > 0
valid_idx = np.where(valid_mask)[0]
sample_idx = np.random.choice(valid_idx, min(10000, len(valid_idx)), replace=False)

X_train = features_flat[sample_idx]
y_train = labels_flat[sample_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Random Forest
start = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_train_time = time.time() - start

# Predict on full image (sample for speed)
sample_pred_idx = np.random.choice(len(features_flat), 50000, replace=False)
X_pred = scaler.transform(features_flat[sample_pred_idx])
rf_pred = rf.predict(X_pred)

# Cross-validation score
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=3, scoring='accuracy')
print(f"RF CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"RF Training time: {rf_train_time:.2f}s")

# Full prediction map
start = time.time()
features_scaled_all = scaler.transform(features_flat)
rf_map_flat = rf.predict(features_scaled_all)
rf_inference_time = time.time() - start
rf_map = rf_map_flat.reshape(h, w)
print(f"RF Inference time: {rf_inference_time:.2f}s")

# ─── Visualization ────────────────────────────────────────────────────────────
colors = ['black', 'darkgreen', 'yellowgreen', 'blue', 'sienna', 'gray']
labels = ['Unclassified', 'Dense Veg', 'Sparse Veg', 'Water', 'Bare Soil', 'Built-up']
cmap = mcolors.ListedColormap(colors)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# RGB reference
rgb = np.stack([data[2], data[1], data[0]], axis=-1)
rgb = np.clip(rgb / 0.3, 0, 1)
axes[0].imshow(rgb)
axes[0].set_title('True Colour RGB\nReference')
axes[0].axis('off')

# Classical map
cmap_list = mcolors.ListedColormap(colors)
im1 = axes[1].imshow(classical_map, cmap=cmap_list, vmin=0, vmax=5)
axes[1].set_title('Classical Rule-Based\nSegmentation')
axes[1].axis('off')
cbar1 = plt.colorbar(im1, ax=axes[1], ticks=range(6))
cbar1.set_ticklabels(labels)

# RF map
im2 = axes[2].imshow(rf_map, cmap=cmap_list, vmin=0, vmax=5)
axes[2].set_title(f'Random Forest Classification\nCV Acc={cv_scores.mean():.3f}')
axes[2].axis('off')
cbar2 = plt.colorbar(im2, ax=axes[2], ticks=range(6))
cbar2.set_ticklabels(labels)

plt.suptitle('Task 4: Land-Cover Classification — Sahiwal, Punjab\nAgriSense CS406 Spring 2026', fontsize=13)
plt.tight_layout()
plt.savefig('data/outputs/task4_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to data/outputs/task4_segmentation.png")

# ─── Class Distribution ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (map_data, title) in zip(axes, [(classical_map, 'Classical'), (rf_map, 'Random Forest')]):
    counts = [np.sum(map_data == i) for i in range(6)]
    ax.bar(labels, counts, color=colors)
    ax.set_title(f'{title} — Class Distribution')
    ax.set_ylabel('Pixel Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/outputs/task4_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Task 4 complete.")