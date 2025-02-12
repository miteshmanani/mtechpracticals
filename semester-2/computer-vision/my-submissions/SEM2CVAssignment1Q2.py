import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_connected_components(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise (optional)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Convert the image to binary using a simple threshold (you can adjust the threshold value)
    _, binary_image = cv2.threshold(
        blurred_image, 128, 255, cv2.THRESH_BINARY_INV)

    # Perform morphological operations to enhance the image (optional)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(
        binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_image = cv2.morphologyEx(
        cleaned_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned_image, connectivity=8)

    # Filter out smaller components (e.g., noise) by area
    min_area = 500  # Minimum area threshold to keep the component
    filtered_labels = np.zeros_like(labels)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_labels[labels == i] = i

    # Assign random colors to each component for visualization
    output_image = np.zeros_like(image)
    for i in range(1, num_labels):
        mask = filtered_labels == i
        output_image[mask] = np.random.randint(
            0, 255, 3)  # Assign random colors

    # Display the original and labeled images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Connected Components')
    plt.axis('off')

    plt.show()


# Path to the uploaded image file
# Update with correct image path
image_path = 'C:\\Users\\mites\\OneDrive\\Desktop\\CV_Assignment 1- database\\Assignment 1 database\\Q2\\tiger.jpg'

# Run the connected components detection
find_connected_components(image_path)
