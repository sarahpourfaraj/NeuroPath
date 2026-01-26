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

        
# تابع رندر
def render(samples_per_pixel):
    image = np.zeros((height, width, 3))
    camera_origin = np.array([0, 0, 0])  # Moved inside the function

    for y in range(height):
        for x in range(width):

            color = np.zeros(3)

            for s in range(samples_per_pixel):
                px = (2 * (x + np.random.rand()) / width - 1)
                py = (1 - 2 * (y + np.random.rand()) / height)

                ray_dir = np.array([px, py, -1])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                t = intersect_sphere(camera_origin, ray_dir,
                                     sphere_center, sphere_radius)

                if t is not None:
                    hit = camera_origin + t * ray_dir
                    normal = hit - sphere_center
                    normal = normal / np.linalg.norm(normal)
                    intensity = max(np.dot(normal, light_dir), 0)
                    sample_color = intensity * sphere_color
                else:
                    sample_color = np.array([0.1, 0.1, 0.1])

                color += sample_color

            image[y, x] = color / samples_per_pixel

    return image


# ----------------------------
# Main execution (only runs when file is executed directly)
# ----------------------------
if __name__ == "__main__":
    image = render(samples_per_pixel=1)

    plt.imshow(np.clip(image, 0, 1))
    plt.axis("off")
    plt.title("Simple Ray Traced Sphere (Noisy)")
    plt.show()