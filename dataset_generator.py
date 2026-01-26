import os
import numpy as np
import matplotlib.pyplot as plt
from raytracer import render

os.makedirs("data/noisy", exist_ok=True)
os.makedirs("data/clean", exist_ok=True)

NUM_IMAGES = 50

for i in range(NUM_IMAGES):
    print(f"Rendering image {i}")

    noisy = render(samples_per_pixel=1)
    clean = render(samples_per_pixel=64)

    plt.imsave(f"data/noisy/img_{i}.png", np.clip(noisy, 0, 1))
    plt.imsave(f"data/clean/img_{i}.png", np.clip(clean, 0, 1))
