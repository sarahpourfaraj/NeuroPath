import os
import matplotlib.pyplot as plt
from raytracer import render

os.makedirs("data/noisy/color", exist_ok=True)
os.makedirs("data/noisy/albedo", exist_ok=True)
os.makedirs("data/noisy/normal", exist_ok=True)

os.makedirs("data/clean/color", exist_ok=True)
os.makedirs("data/clean/albedo", exist_ok=True)
os.makedirs("data/clean/normal", exist_ok=True)

NUM_IMAGES = 50

for i in range(NUM_IMAGES):
    print(f"Rendering {i}")

    for aov in ["color", "albedo", "normal"]:
        noisy = render(samples_per_pixel=1, max_depth=3, aov=aov)
        clean = render(samples_per_pixel=128, max_depth=3, aov=aov)

        plt.imsave(f"data/noisy/{aov}/img_{i}.png", noisy)
        plt.imsave(f"data/clean/{aov}/img_{i}.png", clean)
