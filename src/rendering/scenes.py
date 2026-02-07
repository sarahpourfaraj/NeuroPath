import numpy as np
from path_tracer import Sphere

# Simple scenes (open)
def simple_scene(seed):
    np.random.seed(seed)

    spheres = []
    for _ in range(np.random.randint(1, 3)):
        center = [
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-0.4, 0.4),
            np.random.uniform(-3.0, -5.0),
        ]
        radius = np.random.uniform(0.3, 0.8)
        color = np.random.uniform(0.3, 1.0, size=3)
        spheres.append(Sphere(center, radius, color))

    env_light = np.random.uniform(0.3, 0.6, size=3)
    return spheres, env_light


# Mixed scenes (ground + open)
def mixed_scene(seed):
    np.random.seed(seed)

    spheres = []

    for _ in range(np.random.randint(1, 3)):
        center = [
            np.random.uniform(-0.8, 0.8),
            np.random.uniform(-0.3, 0.5),
            np.random.uniform(-3.0, -5.0),
        ]
        radius = np.random.uniform(0.3, 0.7)
        color = np.random.uniform(0.3, 1.0, size=3)
        spheres.append(Sphere(center, radius, color))

    # ground
    spheres.append(
        Sphere([0, -1001, -4], 1000, [0.8, 0.8, 0.8])
    )

    env_light = np.random.uniform(0.4, 0.7, size=3)
    return spheres, env_light


# Cornell-like scenes (enclosed, GI-heavy)
def cornell_like_scene(seed):
    np.random.seed(seed)

    spheres = []

    # Objects
    spheres.append(
        Sphere(
            [-0.4, -0.3, -3.6],
            0.35,
            [0.9, 0.9, 0.9]
        )
    )

    spheres.append(
        Sphere(
            [0.5, -0.35, -4.2],
            0.45,
            [0.7, 0.7, 0.9]
        )
    )

    # Floor
    spheres.append(Sphere([0, -1001, -4], 1000, [0.9, 0.9, 0.9]))

    # Walls
    spheres.append(Sphere([-1002, 0, -4], 1000, [0.9, 0.3, 0.3]))  # left
    spheres.append(Sphere([1002, 0, -4], 1000, [0.3, 0.9, 0.3]))   # right
    spheres.append(Sphere([0, 0, -1006], 1000, [0.9, 0.9, 0.9]))   # back

    # no ceiling (prevents black images)
    env_light = np.array([1.2, 1.2, 1.2])
    return spheres, env_light


# Scene selector
def random_scene(idx):
    if idx < 240:
        return simple_scene(idx)
    elif idx < 480:
        return mixed_scene(idx)
# Cornell scenes were excluded due to unstable illumination
# and used only in preliminary experiments.
  # else:
  #     return cornell_like_scene(idx)