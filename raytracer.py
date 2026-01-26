import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# تنظیمات تصویر
# ----------------------------
width = 200
height = 200

# ----------------------------
# کره (Sphere)
# ----------------------------
sphere_center = np.array([0, 0, -3])
sphere_radius = 1.0
sphere_color = np.array([1.0, 0.2, 0.2])  # قرمز

# ----------------------------
# نور
# ----------------------------
light_dir = np.array([1, 1, -1])
light_dir = light_dir / np.linalg.norm(light_dir)

# ----------------------------
# تابع برخورد Ray با Sphere
# ----------------------------
def intersect_sphere(ray_origin, ray_dir, center, radius):
    oc = ray_origin - center
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return None

    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    if t < 0:
        return None

    return t

# ----------------------------
# رندر
# ----------------------------
image = np.zeros((height, width, 3))

camera_origin = np.array([0, 0, 0])

for y in range(height):
    for x in range(width):

        # مختصات نرمال‌شده پیکسل
        px = (2 * (x + 0.5) / width - 1)
        py = (1 - 2 * (y + 0.5) / height)

        ray_dir = np.array([px, py, -1])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        t = intersect_sphere(camera_origin, ray_dir,
                             sphere_center, sphere_radius)

        if t is not None:
            hit_point = camera_origin + t * ray_dir
            normal = hit_point - sphere_center
            normal = normal / np.linalg.norm(normal)

            # نور diffuse
            noise = np.random.rand() * 0.5
            intensity = max(np.dot(normal, light_dir), 0) * (1 + noise)

            color = intensity * sphere_color
        else:
            color = np.array([0.1, 0.1, 0.1])  # پس‌زمینه

        image[y, x] = color

# ----------------------------
# نمایش تصویر
# ----------------------------
plt.imshow(np.clip(image, 0, 1))
plt.axis("off")
plt.title("Simple Ray Traced Sphere")
plt.show()
