import numpy as np
import matplotlib.pyplot as plt

WIDTH = 200
HEIGHT = 200


class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)

    def intersect(self, ray_origin, ray_dir):
        oc = ray_origin - self.center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - self.radius * self.radius
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        t = (-b - np.sqrt(disc)) / (2 * a)
        if t < 1e-4:
            return None
        return t


def random_scene():
    spheres = []
    num_spheres = np.random.randint(1, 4)

    for _ in range(num_spheres):
        center = [
            np.random.uniform(-1.2, 1.2),
            np.random.uniform(-0.6, 0.6),
            np.random.uniform(-2.5, -5.0)
        ]
        radius = np.random.uniform(0.4, 1.0)
        color = np.random.uniform(0.2, 1.0, size=3)
        spheres.append(Sphere(center, radius, color))

    env_light = np.random.uniform(0.1, 1.0, size=3)
    return spheres, env_light


def random_hemisphere(normal):
    r1 = np.random.rand()
    r2 = np.random.rand()

    phi = 2 * np.pi * r1
    cos_theta = np.sqrt(1 - r2)
    sin_theta = np.sqrt(r2)

    local = np.array([
        np.cos(phi) * sin_theta,
        np.sin(phi) * sin_theta,
        cos_theta
    ])

    w = normal
    a = np.array([1, 0, 0]) if abs(w[0]) < 0.9 else np.array([0, 1, 0])
    v = np.cross(w, a)
    v /= np.linalg.norm(v)
    u = np.cross(v, w)

    return local[0] * u + local[1] * v + local[2] * w


def intersect_scene(ray_origin, ray_dir, spheres):
    hit_t = np.inf
    hit_obj = None

    for obj in spheres:
        t = obj.intersect(ray_origin, ray_dir)
        if t and t < hit_t:
            hit_t = t
            hit_obj = obj

    if hit_obj is None:
        return None, None

    return hit_obj, hit_t


def radiance(ray_origin, ray_dir, depth, aov, spheres, env_light):
    if depth == 0:
        return np.zeros(3)

    obj, t = intersect_scene(ray_origin, ray_dir, spheres)
    if obj is None:
        return env_light

    hit = ray_origin + t * ray_dir
    normal = hit - obj.center
    normal /= np.linalg.norm(normal)

    if aov == "albedo":
        return obj.color

    if aov == "normal":
        return normal * 0.5 + 0.5

    new_dir = random_hemisphere(normal)
    new_origin = hit + normal * 1e-4

    indirect = radiance(
        new_origin, new_dir,
        depth - 1, aov,
        spheres, env_light
    )

    return obj.color * indirect


def render(samples_per_pixel=1, max_depth=3, aov="color"):
    image = np.zeros((HEIGHT, WIDTH, 3))
    cam_origin = np.array([0, 0, 0])

    spheres, env_light = random_scene()

    for y in range(HEIGHT):
        for x in range(WIDTH):
            col = np.zeros(3)

            for _ in range(samples_per_pixel):
                px = (2 * (x + np.random.rand()) / WIDTH - 1)
                py = (1 - 2 * (y + np.random.rand()) / HEIGHT)

                ray_dir = np.array([px, py, -1])
                ray_dir /= np.linalg.norm(ray_dir)

                col += radiance(
                    cam_origin, ray_dir,
                    max_depth, aov,
                    spheres, env_light
                )

            image[y, x] = col / samples_per_pixel

    return np.clip(image, 0, 1)

def render_with_scene(spheres, env_light, samples_per_pixel=1, max_depth=3, aov="color"):
    image = np.zeros((HEIGHT, WIDTH, 3))
    cam_origin = np.array([0, 0, 0])

    for y in range(HEIGHT):
        for x in range(WIDTH):
            col = np.zeros(3)

            for _ in range(samples_per_pixel):
                px = (2 * (x + np.random.rand()) / WIDTH - 1)
                py = (1 - 2 * (y + np.random.rand()) / HEIGHT)

                ray_dir = np.array([px, py, -1])
                ray_dir /= np.linalg.norm(ray_dir)

                col += radiance(
                    cam_origin, ray_dir,
                    max_depth, aov,
                    spheres, env_light
                )

            image[y, x] = col / samples_per_pixel

    return np.clip(image, 0, 1)


if __name__ == "__main__":
    img = render(samples_per_pixel=8, max_depth=3, aov="color")
    plt.imshow(img)
    plt.axis("off")
    plt.show()
