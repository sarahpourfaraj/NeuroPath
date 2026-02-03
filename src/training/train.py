import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from datasets.denoising_dataset import DenoisingDataset
from models.unet import UNet
from utils.losses import DenoisingLoss

#config
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
VAL_RATIO = 0.1
SAVE_DIR = os.path.join(os.path.dirname(__file__), "../../checkpoints")

os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Dataset + Split
dataset = DenoisingDataset(
    root_dir=os.path.join(os.path.dirname(__file__), "../../data"),
    use_albedo=True,
    use_normal=True,
    use_depth=True
)

val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=1,
    shuffle=False
)

# Model / Loss / Optimizer
model = UNet(in_channels=10, out_channels=3).to(device)
criterion = DenoisingLoss(ssim_weight=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training Loop
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    #Train
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

    #Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"[Epoch {epoch:03d}] "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )

    #save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = os.path.join(SAVE_DIR, "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print("Best model saved")

print("Training + Validation finished.")
