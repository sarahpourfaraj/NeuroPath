import os
import sys
import time
import cv2
import torch
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from rendering.path_tracer import random_scene, render_with_scene
from models.unet import UNet

# Config
WIDTH = 128
HEIGHT = 128
SPP = 1
MAX_DEPTH = 3
DISPLAY_SCALE = 4

DEVICE = "cpu"
CKPT_PATH = os.path.join(PROJECT_DIR, "checkpoints", "unet_aovs_best.pth")

AOVS = ["color", "albedo", "normal", "depth"]

# Load model
print("Loading denoising model on CPU...")
model = UNet(in_channels=10, out_channels=3).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

# Scene
spheres, env_light = random_scene()

#prepare input tensor
def prepare_input(aov_dict):
    """
    aov_dict contains numpy arrays in [0,1]
    returns torch tensor (1,10,H,W)
    """
    color  = torch.from_numpy(aov_dict["color"]).permute(2, 0, 1)
    albedo = torch.from_numpy(aov_dict["albedo"]).permute(2, 0, 1)
    normal = torch.from_numpy(aov_dict["normal"]).permute(2, 0, 1)
    depth  = torch.from_numpy(aov_dict["depth"][..., 0]).unsqueeze(0)

    x = torch.cat([color, albedo, normal, depth], dim=0)
    return x.unsqueeze(0).float().to(DEVICE)

# Demo loop
print("Starting CPU real-time demo (press Q to quit)")

while True:
    t_start = time.time()

    #Render AOVs
    aov_outputs = {}
    for aov in AOVS:
        aov_outputs[aov] = render_with_scene(
            spheres,
            env_light,
            samples_per_pixel=SPP if aov == "color" else 1,
            max_depth=MAX_DEPTH,
            aov=aov
        )

    t_render = time.time()

    #Denoising inference
    x = prepare_input(aov_outputs)

    with torch.no_grad():
        pred = model(x).clamp(0, 1)

    denoised = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    t_denoise = time.time()

    #Visualization
    noisy_vis = (aov_outputs["color"] * 255).astype(np.uint8)
    den_vis   = (denoised * 255).astype(np.uint8)

    combined = np.hstack([noisy_vis, den_vis])
    combined = cv2.resize(
        combined,
        (combined.shape[1] * DISPLAY_SCALE,
         combined.shape[0] * DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST
    )
    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    render_ms = (t_render - t_start) * 1000
    denoise_ms = (t_denoise - t_render) * 1000
    total_ms = (t_denoise - t_start) * 1000
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    cv2.putText(
        combined,
        f"Render: {render_ms:.1f} ms | Denoise: {denoise_ms:.1f} ms | FPS: {fps:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    cv2.putText(
        combined,
        "Left: Noisy (SPP=1) | Right: Denoised (CPU)",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    cv2.namedWindow(
    "CPU Real-Time Ray Tracing Denoising",
    cv2.WINDOW_NORMAL
    )
    cv2.imshow("CPU Real-Time Ray Tracing Denoising", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    
cv2.destroyAllWindows()
print("Demo finished.")
