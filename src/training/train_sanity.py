import os
import sys
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

#reproducibility
torch.manual_seed(42)

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from datasets.denoising_dataset import DenoisingDataset
from models.unet import UNet
from utils.losses import DenoisingLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#sanity: 1 sample
dataset = DenoisingDataset(
    root_dir=os.path.join(os.path.dirname(__file__), "../../data"),
    use_albedo=True,
    use_normal=True,
    use_depth=True
)

sanity_dataset = Subset(dataset, [0])
loader = DataLoader(
    sanity_dataset,
    batch_size=1,
    shuffle=True
)

#load single batch ONCE
x, y = next(iter(loader))
x = x.to(device)
y = y.to(device)

print("Input shape:", x.shape)   # [1, 10, H, W]
print("GT shape:", y.shape)      # [1, 3, H, W]

# Model / Loss / Optimizer
model = UNet(in_channels=10, out_channels=3).to(device)
criterion = DenoisingLoss(ssim_weight=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#Sanity Overfit
num_iters = 500
model.train()

for it in range(num_iters):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 50 == 0:
        print(f"[{it:04d}] loss = {loss.item():.6f}")

print("Sanity overfit finished.")

#to save visual results
model.eval()
with torch.no_grad():
    pred = model(x).clamp(0, 1)

os.makedirs("sanity_outputs", exist_ok=True)

save_image(x[:, :3], "sanity_outputs/input_noisy.png")
save_image(pred, "sanity_outputs/pred_denoised.png")
save_image(y, "sanity_outputs/gt_clean.png")

print("Sanity images saved in sanity_outputs")
