import numpy as np

WIDTH = 200
HEIGHT = 200
EPS = 1e-4


class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.color = np.array(color, dtype=np.float32)

    def intersect(self, ray_origin, ray_dir):
        oc = ray_origin - self.center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - self.radius * self.radius
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        t = (-b - np.sqrt(disc)) / (2 * a)
        if t < EPS:
            return None
        return t


# ----------------------------
# Scene
# ----------------------------
def random_scene():
    spheres = []
    num_spheres = np.random.randint(1, 4)

    for _ in range(num_spheres):
        center = [
            np.random.uniform(-1.2, 1.2),
            np.random.uniform(-0.6, 0.6),
            np.random.uniform(-2.5, -5.0),
        ]
        radius = np.random.uniform(0.4, 1.0)
        color = np.random.uniform(0.2, 1.0, size=3)
        spheres.append(Sphere(center, radius, color))

    env_light = np.random.uniform(0.1, 1.0, size=3)
    return spheres, env_light


# ----------------------------
# Geometry helpers
# ----------------------------
def intersect_scene(ray_origin, ray_dir, spheres):
    hit_t = np.inf
    hit_obj = None
    for obj in spheres:
        t = obj.intersect(ray_origin, ray_dir)
        if t is not None and t < hit_t:
            hit_t = t
            hit_obj = obj
    if hit_obj is None:
        return None, None
    return hit_obj, hit_t


def random_hemisphere(normal):
    r1, r2 = np.random.rand(), np.random.rand()
    phi = 2 * np.pi * r1
    cos_t = np.sqrt(1 - r2)
    sin_t = np.sqrt(r2)

    local = np.array([
        np.cos(phi) * sin_t,
        np.sin(phi) * sin_t,
        cos_t
    ])

    w = normal
    a = np.array([1, 0, 0]) if abs(w[0]) < 0.9 else np.array([0, 1, 0])
    v = np.cross(w, a)
    v /= np.linalg.norm(v)
    u = np.cross(v, w)

    return local[0]*u + local[1]*v + local[2]*w


# ----------------------------
# Radiance
# ----------------------------
def radiance(ray_origin, ray_dir, depth, spheres, env_light):
    if depth == 0:
        return np.zeros(3)

    obj, t = intersect_scene(ray_origin, ray_dir, spheres)
    if obj is None:
        return env_light

    hit = ray_origin + t * ray_dir
    normal = hit - obj.center
    normal /= np.linalg.norm(normal)

    new_dir = random_hemisphere(normal)
    new_origin = hit + normal * EPS

    return obj.color * radiance(
        new_origin, new_dir, depth - 1, spheres, env_light
    )


# ----------------------------
# Render with AOVs
# ----------------------------
def render_with_scene(
    spheres, env_light,
    samples_per_pixel=1,
    max_depth=3,
    aov="color"
):
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    cam_origin = np.array([0, 0, 0], dtype=np.float32)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            col = np.zeros(3)

            for _ in range(samples_per_pixel):
                px = 2 * (x + np.random.rand()) / WIDTH - 1
                py = 1 - 2 * (y + np.random.rand()) / HEIGHT
                ray_dir = np.array([px, py, -1], dtype=np.float32)
                ray_dir /= np.linalg.norm(ray_dir)

                obj, t = intersect_scene(cam_origin, ray_dir, spheres)

                if aov == "depth":
                    sample = 1.0 if t is None else t
                    col += np.array([sample, sample, sample])

                elif aov == "normal":
                    if obj is None:
                        col += 0
                    else:
                        hit = cam_origin + t * ray_dir
                        n = hit - obj.center
                        n /= np.linalg.norm(n)
                        col += n * 0.5 + 0.5

                elif aov == "albedo":
                    col += obj.color if obj is not None else 0

                else:  # color
                    col += radiance(
                        cam_origin, ray_dir,
                        max_depth, spheres, env_light
                    )

            image[y, x] = col / samples_per_pixel

    if aov == "depth":
        image /= image.max() + 1e-8

    return np.clip(image, 0, 1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)
    spheres, env_light = random_scene()

    img = render_with_scene(
        spheres, env_light,
        samples_per_pixel=8,
        max_depth=3,
        aov="color"
    )

    plt.imshow(img)
    plt.axis("off")
    plt.title("Path Tracing Preview")
    plt.show()

