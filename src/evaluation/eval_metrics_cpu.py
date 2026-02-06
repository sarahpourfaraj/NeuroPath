import os
import sys
import time
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2lab, deltaE_ciede2000

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.unet import UNet
from rendering.path_tracer import render_with_scene, random_scene

# Config
DEVICE = "cpu"
WIDTH = 128
HEIGHT = 128
SPP_NOISY = 1
SPP_GT = 64
MAX_DEPTH = 3

NUM_SCENES = 5
FRAMES_PER_SCENE = 10
WARMUP_FRAMES = 3

CKPT_PATH = os.path.join(PROJECT_DIR, "checkpoints", "unet_aovs_best.pth")

# Model
model = UNet(in_channels=10, out_channels=3).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# Stats
psnr_vals, ssim_vals, deltae_vals = [], [], []
render_times, denoise_times = [], []

# Evaluation
with torch.no_grad():
    for s in range(NUM_SCENES):
        spheres, env_light = random_scene()

        # pre-render GT once per scene
        gt = render_with_scene(
            spheres, env_light,
            samples_per_pixel=SPP_GT,
            max_depth=MAX_DEPTH,
            aov="color"
        )

        for f in range(FRAMES_PER_SCENE + WARMUP_FRAMES):
            # Render noisy AOVs
            t0 = time.time()
            color  = render_with_scene(spheres, env_light, SPP_NOISY, MAX_DEPTH, "color")
            albedo = render_with_scene(spheres, env_light, 1, 1, "albedo")
            normal = render_with_scene(spheres, env_light, 1, 1, "normal")
            depth  = render_with_scene(spheres, env_light, 1, 1, "depth")
            t1 = time.time()

            # Prepare input
            x = torch.from_numpy(
                np.concatenate(
                    [color, albedo, normal, depth[..., :1]],
                    axis=-1
                )
            ).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

            # Denoise 
            t2 = time.time()
            pred = model(x).clamp(0, 1)
            t3 = time.time()

            if f < WARMUP_FRAMES:
                continue  # skip warmup

            render_times.append(t1 - t0)
            denoise_times.append(t3 - t2)

            den = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Metrics 
            psnr_vals.append(
                peak_signal_noise_ratio(gt, den, data_range=1.0)
            )
            ssim_vals.append(
                structural_similarity(gt, den, channel_axis=-1, data_range=1.0)
            )

            lab_gt = rgb2lab(gt)
            lab_den = rgb2lab(den)
            deltae_vals.append(
                np.mean(deltaE_ciede2000(lab_gt, lab_den))
            )

# Report
avg_render = np.mean(render_times)
avg_denoise = np.mean(denoise_times)

fps_denoise = 1.0 / avg_denoise
fps_total = 1.0 / (avg_render + avg_denoise)

print("===== CPU Runtime & Quality Evaluation =====")
print(f"PSNR        : {np.mean(psnr_vals):.2f} dB")
print(f"SSIM        : {np.mean(ssim_vals):.4f}")
print(f"ΔE (CIEDE)  : {np.mean(deltae_vals):.2f}")
print("--------------------------------------------")
print(f"Render time : {avg_render*1000:.1f} ms")
print(f"Denoise time: {avg_denoise*1000:.1f} ms")
print(f"FPS (NN)    : {fps_denoise:.2f}")
print(f"FPS (Total) : {fps_total:.2f}")
