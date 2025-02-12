import numpy as np
import cv2
import matplotlib.pyplot as plt

# Input grayscale image
image = np.array([
    [12, 56, 78, 45, 90],
    [34, 67, 89, 12, 40],
    [23, 89, 67, 45, 56],
    [78, 45, 12, 89, 23],
    [90, 12, 67, 56, 34]
], dtype=np.uint8)

# Threshold value
T = 50

# Apply global thresholding
_, binary_image = cv2.threshold(image, T, 1, cv2.THRESH_BINARY)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Binary Image (Thresholded)")
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.show()
