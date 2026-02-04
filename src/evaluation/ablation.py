import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

#reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "ablation")

os.makedirs(RESULTS_DIR, exist_ok=True)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from datasets.denoising_dataset import DenoisingDataset
from models.unet import UNet
from utils.losses import DenoisingLoss
from utils.metrics import psnr, ssim

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#ablation configurations
configs = {
    "color_only":   {"use_albedo": False, "use_normal": False, "use_depth": False},
    "color_albedo": {"use_albedo": True,  "use_normal": False, "use_depth": False},
    "color_normal": {"use_albedo": False, "use_normal": True,  "use_depth": False},
    "color_depth":  {"use_albedo": False, "use_normal": False, "use_depth": True},
    "full":         {"use_albedo": True,  "use_normal": True,  "use_depth": True},
}

# Load validation indices (from baseline training)
val_idx_path = os.path.join(CKPT_DIR, "val_indices.json")
assert os.path.exists(val_idx_path), "val_indices.json not found. Run train.py first."

with open(val_idx_path, "r") as f:
    val_indices = set(json.load(f))

# Ablation loop
for cfg_name, cfg in configs.items():
    print(f"\n=== Ablation: {cfg_name} ===")

    # Dataset for this configuration
    dataset = DenoisingDataset(
        root_dir=os.path.join(PROJECT_DIR, "data"),
        use_albedo=cfg["use_albedo"],
        use_normal=cfg["use_normal"],
        use_depth=cfg["use_depth"]
    )

    # Derive train indices (dataset − val)
    all_indices = set(range(len(dataset)))
    train_indices = sorted(list(all_indices - val_indices))
    val_indices_sorted = sorted(list(val_indices))

    print(f"Train samples: {len(train_indices)} | Val samples: {len(val_indices_sorted)}")

    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices_sorted)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Input channels
    in_channels = 3
    in_channels += 3 if cfg["use_albedo"] else 0
    in_channels += 3 if cfg["use_normal"] else 0
    in_channels += 1 if cfg["use_depth"] else 0

    # Model / Loss / Optimizer
    model = UNet(in_channels=in_channels, out_channels=3).to(device)
    criterion = DenoisingLoss(ssim_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    best_val = float("inf")

    # Training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(CKPT_DIR, f"{cfg_name}_best.pth")
            )

    # Evaluation
    model.eval()
    combo_dir = os.path.join(RESULTS_DIR, cfg_name)
    img_dir = os.path.join(combo_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    psnr_vals, ssim_vals = [], []

    with torch.no_grad():
        for idx, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x).clamp(0, 1)

            psnr_vals.append(psnr(pred, y).item())
            ssim_vals.append(ssim(pred, y).item())

            save_image(x[:, :3], os.path.join(img_dir, f"{idx:03d}_noisy.png"))
            save_image(pred,     os.path.join(img_dir, f"{idx:03d}_pred.png"))
            save_image(y,        os.path.join(img_dir, f"{idx:03d}_gt.png"))

    # Metrics
    psnr_mean, psnr_std = np.mean(psnr_vals), np.std(psnr_vals)
    ssim_mean, ssim_std = np.mean(ssim_vals), np.std(ssim_vals)

    with open(os.path.join(combo_dir, "metrics.txt"), "w") as f:
        f.write(f"PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}\n")
        f.write(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}\n")

    print(
        f"{cfg_name} | "
        f"PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} | "
        f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}"
    )
