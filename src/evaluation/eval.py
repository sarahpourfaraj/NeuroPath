import os
import sys
import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "eval")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
    
os.makedirs(RESULTS_DIR, exist_ok=True)

from datasets.denoising_dataset import DenoisingDataset
from models.unet import UNet
from utils.metrics import psnr, ssim

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#load dataset (val or test)(full)
dataset = DenoisingDataset(
    root_dir=os.path.join(PROJECT_DIR, "data"),
    use_albedo=True,
    use_normal=True,
    use_depth=True
)

#validation indices: 90% train / 10% val
val_idx_path = os.path.join(CKPT_DIR, "val_indices.json")
assert os.path.exists(val_idx_path), (
    "val_indices.json not found. "
    "You must run train.py first."
)

with open(val_idx_path, "r") as f:
    val_indices = json.load(f)

val_dataset = Subset(dataset, val_indices)

loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)
print(f"Evaluating on {len(val_dataset)} validation samples")

#load model
model = UNet(in_channels=10, out_channels=3).to(device)
ckpt_path = os.path.join(PROJECT_DIR, "checkpoints", "unet_aovs_best.pth")
assert os.path.exists(ckpt_path), "Checkpoint not found."

model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

#output dirs
img_dir = os.path.join(RESULTS_DIR, "images")
os.makedirs(img_dir, exist_ok=True)

# Evaluation
psnr_vals = []
ssim_vals = []

with torch.no_grad():
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        pred = model(x).clamp(0, 1)

        psnr_vals.append(psnr(pred, y).item())
        ssim_vals.append(ssim(pred, y).item())

        #save qualitative results (ALL validation samples)
        save_image(pred, os.path.join(img_dir, f"{idx:03d}_pred.png"))
        save_image(y,    os.path.join(img_dir, f"{idx:03d}_gt.png"))

# Report
mean_psnr = sum(psnr_vals) / len(psnr_vals)
mean_ssim = sum(ssim_vals) / len(ssim_vals)

print(f"PSNR: {mean_psnr:.2f}")
print(f"SSIM: {mean_ssim:.4f}")

metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"PSNR: {mean_psnr:.2f}\n")
    f.write(f"SSIM: {mean_ssim:.4f}\n")
