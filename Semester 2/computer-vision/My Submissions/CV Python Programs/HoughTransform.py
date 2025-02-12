# C:\\Users\\mites\\OneDrive\Desktop\\13549827_web1_180914-CDT-booknotes.jpg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def hough_transform(image_path):
    # Open and convert the image to grayscale if it isn't already
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # Binarize the image - make it black and white
    threshold = 128
    edges = np.where(img_array < threshold, 0, 1)

    # Hough space parameters
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = edges.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Create the Hough accumulator
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)

    # Vote in the Hough accumulator
    for y in range(height):  # Ensure we don't go over the height
        for x in range(width):  # Ensure we don't go over the width
            if edges[y, x] > 0:
                for theta_idx, theta in enumerate(thetas):
                    rho = int(x * np.cos(theta) + y * np.sin(theta))
                    accumulator[rho + diag_len, theta_idx] += 1

    # Find peaks in the accumulator - this is a simple approach
    threshold = 150  # Adjust based on image and expected lines
    lines = np.argwhere(accumulator >= threshold)

    # Convert back to image coordinates
    detected_lines = []
    for rho_idx, theta_idx in lines:
        rho = rhos[rho_idx - diag_len]
        theta = thetas[theta_idx]
        detected_lines.append((rho, theta))

    return detected_lines, edges


# Example usage:
# Replace with your actual image path
image_path = r"C:\Users\mites\OneDrive\Desktop\The-famous-Lena-image-often-used-as-an-example-in-image-processing.png"
detected_lines, edges, width, height = hough_transform(image_path)

# Plotting the results
plt.imshow(edges, cmap='gray')
for rho, theta in detected_lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # Clip the line to the image bounds
    x1, x2 = np.clip([x1, x2], 0, width - 1)
    y1, y2 = np.clip([y1, y2], 0, height - 1)

    plt.plot((x1, x2), (y1, y2), 'r-')
plt.show()
