"""
AgriSense - Task 2: Weather-Effect Analysis and Image Restoration
CS406 - Digital Image Processing, Spring 2026
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio
import warnings
warnings.filterwarnings('ignore')

# ─── Load Season 1 Composite ──────────────────────────────────────────────────
tif_path = 'data/raw/S2_Season1_BOA_Composite.tif'

with rasterio.open(tif_path) as src:
    data = src.read()

# Use RGB bands (B4=index2, B3=index1, B2=index0)
rgb = np.stack([data[2], data[1], data[0]], axis=-1)
rgb = np.nan_to_num(rgb, nan=0.0)
rgb = np.clip(rgb / 0.3, 0, 1)
rgb_uint8 = (rgb * 255).astype(np.uint8)

# ─── 1. Synthetic Haze Addition ───────────────────────────────────────────────
def add_haze(image, haze_level=0.4):
    haze = np.ones_like(image, dtype=np.float32) * 255 * haze_level
    hazed = image.astype(np.float32) * (1 - haze_level) + haze
    return np.clip(hazed, 0, 255).astype(np.uint8)

hazed_rgb = add_haze(rgb_uint8, haze_level=0.4)

# ─── 2. Dark Channel Prior (DCP) Haze Removal ─────────────────────────────────
def get_dark_channel(image, patch_size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel, top_percent=0.001):
    num_pixels = dark_channel.size
    num_brightest = max(1, int(num_pixels * top_percent))
    flat_dark = dark_channel.flatten()
    indices = np.argsort(flat_dark)[-num_brightest:]
    rows, cols = np.unravel_index(indices, dark_channel.shape)
    atmospheric = np.mean(image[rows, cols], axis=0)
    return atmospheric

def estimate_transmission(image, atmospheric, patch_size=15, omega=0.95):
    normalized = image.astype(np.float64) / atmospheric
    dark = get_dark_channel(normalized, patch_size)
    transmission = 1 - omega * dark
    return np.clip(transmission, 0.1, 1.0)

def recover_scene(image, transmission, atmospheric, t0=0.1):
    transmission = np.maximum(transmission, t0)
    recovered = np.zeros_like(image, dtype=np.float64)
    for c in range(3):
        recovered[:, :, c] = (image[:, :, c].astype(np.float64) - atmospheric[c]) / transmission + atmospheric[c]
    return np.clip(recovered, 0, 255).astype(np.uint8)

def dcp_dehaze(image):
    img_float = image.astype(np.float64)
    dark = get_dark_channel(image)
    atmospheric = estimate_atmospheric_light(image, dark)
    transmission = estimate_transmission(img_float, atmospheric)
    recovered = recover_scene(img_float, transmission, atmospheric)
    return recovered

print("Applying DCP haze removal...")
dehazed_dcp = dcp_dehaze(hazed_rgb)
print("DCP done")

# ─── 3. CLAHE Fog Enhancement ─────────────────────────────────────────────────
def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    channels = [clahe.apply(image[:, :, c]) for c in range(3)]
    return np.stack(channels, axis=-1)

clahe_enhanced = apply_clahe(rgb_uint8)
print("CLAHE done")

# ─── 4. FFT Rain Streak Removal ───────────────────────────────────────────────
def remove_rain_streaks_fft(image):
    result = np.zeros_like(image)
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        # Suppress horizontal streaks (vertical frequencies)
        mask = np.ones((rows, cols), np.float32)
        mask[crow-2:crow+2, :] = 0
        mask[crow-2:crow+2, ccol-30:ccol+30] = 1  # keep DC component
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        filtered = np.abs(np.fft.ifft2(f_ishift))
        result[:, :, c] = np.clip(filtered, 0, 255).astype(np.uint8)
    return result

fft_filtered = remove_rain_streaks_fft(rgb_uint8)
print("FFT rain streak removal done")

# ─── 5. Compute PSNR and SSIM ─────────────────────────────────────────────────
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

psnr_haze    = psnr(rgb_uint8, hazed_rgb)
psnr_dcp     = psnr(rgb_uint8, dehazed_dcp)
psnr_clahe   = psnr(rgb_uint8, clahe_enhanced)

ssim_haze  = ssim(rgb_uint8, hazed_rgb,   channel_axis=2)
ssim_dcp   = ssim(rgb_uint8, dehazed_dcp, channel_axis=2)
ssim_clahe = ssim(rgb_uint8, clahe_enhanced, channel_axis=2)

print("\n=== Evaluation Metrics ===")
print(f"{'Method':<20} {'PSNR (dB)':<15} {'SSIM':<10}")
print(f"{'Hazed Input':<20} {psnr_haze:<15.2f} {ssim_haze:<10.4f}")
print(f"{'DCP Dehazed':<20} {psnr_dcp:<15.2f} {ssim_dcp:<10.4f}")
print(f"{'CLAHE Enhanced':<20} {psnr_clahe:<15.2f} {ssim_clahe:<10.4f}")

# ─── 6. Visualization ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0,0].imshow(rgb_uint8);         axes[0,0].set_title('Original BOA');        axes[0,0].axis('off')
axes[0,1].imshow(hazed_rgb);         axes[0,1].set_title('Synthetic Haze');       axes[0,1].axis('off')
axes[0,2].imshow(dehazed_dcp);       axes[0,2].set_title(f'DCP Dehazed\nPSNR={psnr_dcp:.1f}dB SSIM={ssim_dcp:.3f}'); axes[0,2].axis('off')
axes[1,0].imshow(clahe_enhanced);    axes[1,0].set_title(f'CLAHE Enhanced\nPSNR={psnr_clahe:.1f}dB SSIM={ssim_clahe:.3f}'); axes[1,0].axis('off')
axes[1,1].imshow(fft_filtered);      axes[1,1].set_title('FFT Rain Streak Filter'); axes[1,1].axis('off')

# Metrics bar chart
methods = ['Hazed', 'DCP', 'CLAHE']
psnr_vals = [psnr_haze, psnr_dcp, psnr_clahe]
axes[1,2].bar(methods, psnr_vals, color=['red','green','blue'], alpha=0.7)
axes[1,2].set_ylabel('PSNR (dB)')
axes[1,2].set_title('PSNR Comparison')
axes[1,2].grid(axis='y', alpha=0.3)

plt.suptitle('Task 2: Weather-Effect Analysis and Image Restoration\nAgriSense CS406', fontsize=13)
plt.tight_layout()
plt.savefig('data/outputs/task2_restoration.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved to data/outputs/task2_restoration.png")
print("Task 2 complete.")