import os
import torch
from torch.utils.data import Dataset
import imageio.v2 as imageio
import numpy as np


class DenoisingDataset(Dataset):
    def __init__(
        self,
        root_dir,
        use_albedo=True,
        use_normal=True,
        use_depth=True
    ):
        self.root = root_dir

        self.use_albedo = use_albedo
        self.use_normal = use_normal
        self.use_depth = use_depth

        self.noisy_color_dir = os.path.join(root_dir, "noisy/color")
        self.clean_color_dir = os.path.join(root_dir, "clean/color")

        self.noisy_albedo_dir = os.path.join(root_dir, "noisy/albedo")
        self.noisy_normal_dir = os.path.join(root_dir, "noisy/normal")
        self.noisy_depth_dir = os.path.join(root_dir, "noisy/depth")

        self.filenames = sorted(os.listdir(self.noisy_color_dir))

    def __len__(self):
        return len(self.filenames)

    def load_rgb(self, path):
        img = imageio.imread(path).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)

    def load_depth(self, path):
        img = imageio.imread(path).astype(np.float32) / 255.0
        img = img[..., 0]
        return torch.from_numpy(img).unsqueeze(0)  # (1,H,W)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        inputs = []

        # noisy color
        inputs.append(
            self.load_rgb(os.path.join(self.noisy_color_dir, name))
        )

        if self.use_albedo:
            inputs.append(
                self.load_rgb(os.path.join(self.noisy_albedo_dir, name))
            )

        if self.use_normal:
            inputs.append(
                self.load_rgb(os.path.join(self.noisy_normal_dir, name))
            )

        if self.use_depth:
            inputs.append(
                self.load_depth(os.path.join(self.noisy_depth_dir, name))
            )

        x = torch.cat(inputs, dim=0)   # (10,H,W)

        y = self.load_rgb(
            os.path.join(self.clean_color_dir, name)
        )

        return x, y
    
#to test    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "../../data")

dataset = DenoisingDataset(ROOT_DIR)
x, y = dataset[0]
print(x.shape, y.shape)
