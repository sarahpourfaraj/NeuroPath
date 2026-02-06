import numpy as np

WIDTH = 200
HEIGHT = 200
EPS = 1e-4
MAX_DEPTH_VAL = 10.0

# Geometry
class Sphere:
    def __init__(self, center, radius, color, roughness=1.0):
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.color = np.array(color, dtype=np.float32)
        self.roughness = roughness

    def intersect(self, ray_origin, ray_dir):
        oc = ray_origin - self.center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - self.radius * self.radius
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        t = (-b - np.sqrt(disc)) / (2 * a)
        return t if t > EPS else None


class Box:
    def __init__(self, min_corner, max_corner, color, roughness=1.0):
        self.min = np.array(min_corner, dtype=np.float32)
        self.max = np.array(max_corner, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.roughness = roughness

    def intersect(self, ray_origin, ray_dir):
        inv = 1.0 / (ray_dir + 1e-8)
        tmin = (self.min - ray_origin) * inv
        tmax = (self.max - ray_origin) * inv

        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)

        t_near = np.max(t1)
        t_far = np.min(t2)

        if t_near > t_far or t_far < EPS:
            return None
        return t_near


class Plane:
    def __init__(self, y, color):
        self.y = y
        self.color = np.array(color, dtype=np.float32)
        self.roughness = 1.0

    def intersect(self, ray_origin, ray_dir):
        if abs(ray_dir[1]) < EPS:
            return None
        t = (self.y - ray_origin[1]) / ray_dir[1]
        return t if t > EPS else None


# Helpers
def intersect_scene(ray_origin, ray_dir, objects):
    hit_t = np.inf
    hit_obj = None
    for obj in objects:
        t = obj.intersect(ray_origin, ray_dir)
        if t is not None and t < hit_t:
            hit_t = t
            hit_obj = obj
    return hit_obj, hit_t


def compute_normal(obj, hit):
    if isinstance(obj, Sphere):
        n = hit - obj.center
        return n / np.linalg.norm(n)

    if isinstance(obj, Plane):
        return np.array([0, 1, 0], dtype=np.float32)

    if isinstance(obj, Box):
        eps = 1e-4
        n = np.zeros(3)
        for i in range(3):
            if abs(hit[i] - obj.min[i]) < eps:
                n[i] = -1
            elif abs(hit[i] - obj.max[i]) < eps:
                n[i] = 1
        return n / (np.linalg.norm(n) + 1e-8)


def random_hemisphere(normal, roughness=1.0):
    r1, r2 = np.random.rand(), np.random.rand()
    phi = 2 * np.pi * r1
    cos_t = (1 - r2) ** (1 / (roughness + 1))
    sin_t = np.sqrt(1 - cos_t * cos_t)

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

    return local[0] * u + local[1] * v + local[2] * w


def radiance(ray_origin, ray_dir, depth, objects, env_light):
    if depth == 0:
        return np.zeros(3)

    obj, t = intersect_scene(ray_origin, ray_dir, objects)
    if obj is None:
        return env_light

    hit = ray_origin + t * ray_dir
    normal = compute_normal(obj, hit)

    new_dir = random_hemisphere(normal, obj.roughness)
    new_origin = hit + normal * EPS

    return obj.color * radiance(
        new_origin, new_dir, depth - 1, objects, env_light
    )


# Scene Generator
def random_scene(stage):
    objects = []

    if stage == 1:  # spheres only
        for _ in range(np.random.randint(1, 4)):
            objects.append(
                Sphere(
                    center=[
                        np.random.uniform(-1, 1),
                        np.random.uniform(-0.5, 0.5),
                        np.random.uniform(-3, -6)
                    ],
                    radius=np.random.uniform(0.4, 0.9),
                    color=np.random.uniform(0.3, 1.0, 3)
                )
            )

    elif stage == 2:  # boxes
        objects.append(Plane(-1.0, [0.8, 0.8, 0.8]))
        for _ in range(2):
            objects.append(
                Box(
                    [-1.0, -1.0, -6.0],
                    [1.0, np.random.uniform(-0.2, 0.6), -4.0],
                    np.random.uniform(0.3, 1.0, 3)
                )
            )

    else:  # mixed
        objects.append(Plane(-1.0, [0.7, 0.7, 0.7]))
        for _ in range(np.random.randint(2, 5)):
            if np.random.rand() < 0.5:
                objects.append(
                    Sphere(
                        center=[
                            np.random.uniform(-1, 1),
                            np.random.uniform(-0.5, 0.5),
                            np.random.uniform(-3, -6)
                        ],
                        radius=np.random.uniform(0.3, 0.8),
                        color=np.random.uniform(0.3, 1.0, 3)
                    )
                )
            else:
                objects.append(
                    Box(
                        [-1.0, -1.0, -6.0],
                        [1.0, np.random.uniform(-0.2, 0.6), -4.0],
                        np.random.uniform(0.3, 1.0, 3)
                    )
                )

    env_light = np.random.uniform(0.2, 0.7, 3)
    return objects, env_light


# Renderer
def render_with_scene(objects, env_light, samples_per_pixel=1, max_depth=3, aov="color"):
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

                obj, t = intersect_scene(cam_origin, ray_dir, objects)

                if aov == "depth":
                    d = MAX_DEPTH_VAL if t is None else min(t, MAX_DEPTH_VAL)
                    col += d

                elif aov == "normal" and obj is not None:
                    hit = cam_origin + t * ray_dir
                    n = compute_normal(obj, hit)
                    col += n * 0.5 + 0.5

                elif aov == "albedo" and obj is not None:
                    col += obj.color

                elif aov == "color":
                    col += radiance(cam_origin, ray_dir, max_depth, objects, env_light)

            image[y, x] = col / samples_per_pixel

    if aov == "depth":
        image /= image.max() + 1e-8
        image = np.repeat(image[..., None], 3, axis=2)

    return np.clip(image, 0, 1)
