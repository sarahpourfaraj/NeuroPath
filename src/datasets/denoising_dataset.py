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
        """
        Args:
            root_dir (str): path to data/
            use_* (bool): which AOVs to include as input
        """

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

    def load_image(self, path):
        img = imageio.imread(path).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
        return img

    def __getitem__(self, idx):
        name = self.filenames[idx]

        #noisy color
        noisy_color = self.load_image(
            os.path.join(self.noisy_color_dir, name)
        )

        inputs = [noisy_color]

        #optional AOVs
        if self.use_albedo:
            albedo = self.load_image(
                os.path.join(self.noisy_albedo_dir, name)
            )
            inputs.append(albedo)

        if self.use_normal:
            normal = self.load_image(
                os.path.join(self.noisy_normal_dir, name)
            )
            inputs.append(normal)

        if self.use_depth:
            depth = self.load_image(
                os.path.join(self.noisy_depth_dir, name)
            )
            inputs.append(depth)

        #stack input channels
        x = torch.cat(inputs, dim=0)

        #ground truth
        y = self.load_image(
            os.path.join(self.clean_color_dir, name)
        )

        return x, y
