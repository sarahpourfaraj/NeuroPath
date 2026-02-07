import os
import numpy as np
import imageio

from path_tracer import render_with_scene
from scenes import random_scene

NOISY_AOVS = ["color", "albedo", "normal", "depth"]
NUM_IMAGES = 480 # simple + mixed only

for aov in NOISY_AOVS:
    os.makedirs(f"../data/noisy/{aov}", exist_ok=True)

os.makedirs("../data/clean/color", exist_ok=True)

for i in range(NUM_IMAGES):
    print(f"Rendering scene {i}")
    spheres, env_light = random_scene(i)

    # noisy: all AOVs
    for aov in NOISY_AOVS:
        noisy = render_with_scene(
            spheres, env_light,
            samples_per_pixel=1,
            max_depth=3,
            aov=aov
        )

        imageio.imwrite(
            f"../data/noisy/{aov}/img_{i}.png",
            (noisy * 255).astype(np.uint8)
        )

    # clean: jsut color
    clean = render_with_scene(
        spheres, env_light,
        samples_per_pixel=128,
        max_depth=3,
        aov="color"
    )

    imageio.imwrite(
        f"../data/clean/color/img_{i}.png",
        (clean * 255).astype(np.uint8)
    )
