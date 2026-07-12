"""
AgriSense - Task 6: Super-Resolution Fusion and Compression (Bonus)
CS406 - Digital Image Processing, Spring 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pywt
warnings.filterwarnings('ignore')

# ─── Load Data ────────────────────────────────────────────────────────────────
tif_path = 'data/raw/S2_Season1_BOA_Composite.tif'
with rasterio.open(tif_path) as src:
    data = src.read().astype(np.float32)

data = np.nan_to_num(data, nan=0.0)

# Use NDVI band (B8-B4) as single channel for SR demo
B4 = data[2]
B8 = data[6]
ndvi = np.where((B8 + B4) != 0, (B8 - B4) / (B8 + B4), 0)
ndvi = np.clip(ndvi, -1, 1)

# Normalize to 0-1
ndvi_norm = (ndvi + 1) / 2

print(f"Original NDVI shape: {ndvi_norm.shape}")
print(f"Value range: {ndvi_norm.min():.4f} - {ndvi_norm.max():.4f}")

# ─── Simulate Low Resolution (30m Landsat equivalent) ────────────────────────
from skimage.transform import resize

scale_factor = 3  # 10m → 30m simulation
h, w = ndvi_norm.shape
h_lr = h // scale_factor
w_lr = w // scale_factor

# Downsample to LR
ndvi_lr = resize(ndvi_norm, (h_lr, w_lr), anti_aliasing=True)
# Upsample back to original size (bicubic baseline)
ndvi_bicubic = resize(ndvi_lr, (h, w), order=3, anti_aliasing=True)

print(f"LR shape: {ndvi_lr.shape}")
print(f"HR shape: {ndvi_norm.shape}")

# ─── SRCNN Architecture ───────────────────────────────────────────────────────
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1,  kernel_size=5, padding=2)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# ─── Prepare Training Patches ─────────────────────────────────────────────────
def extract_patches(hr_img, lr_upscaled, patch_size=32, stride=32, max_patches=500):
    hr_patches, lr_patches = [], []
    h, w = hr_img.shape
    for i in range(0, h - patch_size, stride):
        for j in range(0, w - patch_size, stride):
            hr_patch = hr_img[i:i+patch_size, j:j+patch_size]
            lr_patch = lr_upscaled[i:i+patch_size, j:j+patch_size]
            hr_patches.append(hr_patch)
            lr_patches.append(lr_patch)
            if len(hr_patches) >= max_patches:
                return np.array(hr_patches), np.array(lr_patches)
    return np.array(hr_patches), np.array(lr_patches)

print("\nExtracting patches for SRCNN training...")
hr_patches, lr_patches = extract_patches(ndvi_norm, ndvi_bicubic, max_patches=500)

# Convert to tensors
hr_tensor = torch.FloatTensor(hr_patches).unsqueeze(1)
lr_tensor = torch.FloatTensor(lr_patches).unsqueeze(1)

dataset = TensorDataset(lr_tensor, hr_tensor)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

# ─── Train SRCNN ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training SRCNN...")
start = time.time()
losses = []
for epoch in range(20):
    epoch_loss = 0
    for lr_batch, hr_batch in loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)
        optimizer.zero_grad()
        output = model(lr_batch)
        loss = criterion(output, hr_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(loader)
    losses.append(avg_loss)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/20 — Loss: {avg_loss:.6f}")

train_time = time.time() - start
print(f"Training complete in {train_time:.1f}s")

# ─── Inference ────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    # Process in tiles to avoid memory issues
    tile_size = 256
    srcnn_output = np.zeros_like(ndvi_norm)
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = ndvi_bicubic[i:i+tile_size, j:j+tile_size]
            tile_t = torch.FloatTensor(tile).unsqueeze(0).unsqueeze(0).to(device)
            out = model(tile_t).squeeze().cpu().numpy()
            srcnn_output[i:i+out.shape[0], j:j+out.shape[1]] = out

srcnn_output = np.clip(srcnn_output, 0, 1)

# ─── Metrics ──────────────────────────────────────────────────────────────────
psnr_bicubic = psnr(ndvi_norm, ndvi_bicubic, data_range=1.0)
psnr_srcnn   = psnr(ndvi_norm, srcnn_output,  data_range=1.0)
ssim_bicubic = ssim(ndvi_norm, ndvi_bicubic,  data_range=1.0)
ssim_srcnn   = ssim(ndvi_norm, srcnn_output,  data_range=1.0)

print(f"\n=== Super-Resolution Metrics ===")
print(f"{'Method':<15} {'PSNR (dB)':<12} {'SSIM':<10}")
print(f"{'Bicubic':<15} {psnr_bicubic:<12.2f} {ssim_bicubic:<10.4f}")
print(f"{'SRCNN':<15} {psnr_srcnn:<12.2f} {ssim_srcnn:<10.4f}")

# ─── Wavelet Compression ──────────────────────────────────────────────────────
print("\n--- Wavelet Compression ---")
import pywt

def wavelet_compress(image, wavelet='db4', level=3, keep_ratio=0.1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    # Zero out small coefficients
    threshold = np.percentile(np.abs(coeffs[0]), (1 - keep_ratio) * 100)
    new_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        new_coeffs.append(tuple(
            np.where(np.abs(d) > threshold, d, 0) for d in detail
        ))
    reconstructed = pywt.waverec2(new_coeffs, wavelet)
    return reconstructed[:image.shape[0], :image.shape[1]]

compressed_10  = wavelet_compress(ndvi_norm, keep_ratio=0.10)
compressed_25  = wavelet_compress(ndvi_norm, keep_ratio=0.25)

psnr_c10 = psnr(ndvi_norm, compressed_10, data_range=1.0)
psnr_c25 = psnr(ndvi_norm, compressed_25, data_range=1.0)
ssim_c10 = ssim(ndvi_norm, compressed_10, data_range=1.0)
ssim_c25 = ssim(ndvi_norm, compressed_25, data_range=1.0)

print(f"{'Wavelet 10%':<15} {psnr_c10:<12.2f} {ssim_c10:<10.4f}")
print(f"{'Wavelet 25%':<15} {psnr_c25:<12.2f} {ssim_c25:<10.4f}")

# ─── Visualization ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0,0].imshow(ndvi_norm,    cmap='RdYlGn', vmin=0, vmax=1)
axes[0,0].set_title('Original HR NDVI\n(10m Sentinel-2)'); axes[0,0].axis('off')

axes[0,1].imshow(ndvi_bicubic, cmap='RdYlGn', vmin=0, vmax=1)
axes[0,1].set_title(f'Bicubic Upscale (30m→10m)\nPSNR={psnr_bicubic:.1f}dB SSIM={ssim_bicubic:.3f}'); axes[0,1].axis('off')

axes[0,2].imshow(srcnn_output, cmap='RdYlGn', vmin=0, vmax=1)
axes[0,2].set_title(f'SRCNN Super-Resolution\nPSNR={psnr_srcnn:.1f}dB SSIM={ssim_srcnn:.3f}'); axes[0,2].axis('off')

axes[1,0].imshow(compressed_10, cmap='RdYlGn', vmin=0, vmax=1)
axes[1,0].set_title(f'Wavelet Compression 10%\nPSNR={psnr_c10:.1f}dB SSIM={ssim_c10:.3f}'); axes[1,0].axis('off')

axes[1,1].imshow(compressed_25, cmap='RdYlGn', vmin=0, vmax=1)
axes[1,1].set_title(f'Wavelet Compression 25%\nPSNR={psnr_c25:.1f}dB SSIM={ssim_c25:.3f}'); axes[1,1].axis('off')

# Training loss curve
axes[1,2].plot(range(1, 21), losses, 'b-o', markersize=4)
axes[1,2].set_xlabel('Epoch')
axes[1,2].set_ylabel('MSE Loss')
axes[1,2].set_title('SRCNN Training Loss Curve')
axes[1,2].grid(alpha=0.3)

plt.suptitle('Task 6: Super-Resolution Fusion and Wavelet Compression\nAgriSense CS406 Spring 2026', fontsize=13)
plt.tight_layout()
plt.savefig('data/outputs/task6_superres.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved to data/outputs/task6_superres.png")
print("Task 6 complete.")