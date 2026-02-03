import os
import sys
import json
import torch
from torch.utils.data import DataLoader, random_split

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

os.makedirs(CKPT_DIR, exist_ok=True)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from datasets.denoising_dataset import DenoisingDataset
from models.unet import UNet
from utils.losses import DenoisingLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = DenoisingDataset(
        root_dir=os.path.join(PROJECT_DIR, "data"),
        use_albedo=True,
        use_normal=True,
        use_depth=True
    )

    # 90% train / 10% val
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )

    val_idx_path = os.path.join(CKPT_DIR, "val_indices.json")
    with open(val_idx_path, "w") as f:
        json.dump(val_ds.indices, f)

    print(f"Saved {len(val_ds)} validation indices to {val_idx_path}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,   # Windows-safe
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Model / Loss / Optimizer
    model = UNet(in_channels=10, out_channels=3).to(device)
    criterion = DenoisingLoss(ssim_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    num_epochs = 30

    # Training loop
    for epoch in range(num_epochs):
        #train
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        #validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1:03d}] | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

        #save best model
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(CKPT_DIR, "unet_aovs_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print("Best model saved")

    print("Training finished.")


#entry point
if __name__ == "__main__":
    main()
