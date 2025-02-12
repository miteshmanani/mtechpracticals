import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def kmeans_quantization(image, k):
    """
    Apply k-Means quantization to an image.

    Parameters:
    - image: Original image as a NumPy array.
    - k: Number of clusters for k-Means.

    Returns:
    - quantized_image: Quantized image.
    """
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Apply k-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # Create the quantized image
    quantized_image = centers[labels].reshape(image.shape)

    return quantized_image


def calculate_mse(original_image, quantized_image):
    """
    Calculate the Mean Squared Error (MSE) between the original and quantized images.

    Parameters:
    - original_image: Original image as a NumPy array.
    - quantized_image: Quantized image.

    Returns:
    - mse: Mean Squared Error.
    """
    # Flatten the images to 2D arrays
    original_flat = original_image.reshape(-1, 3)
    quantized_flat = quantized_image.reshape(-1, 3)

    mse = mean_squared_error(original_flat, quantized_flat)
    return mse


def main():
    # Read the RGB image as input
    # Replace with your image path
    image_path = 'C:\\Users\\mites\\OneDrive\\Desktop\\CV_Assignment 1- database\\Assignment 1 database\\Q1\\RGB image 4.jpg'
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Varying k values for k-Means
    k_values = [1, 2, 4, 8, 16]

    # Store MSE values for analysis
    mse_values = []

    # Perform k-Means quantization for each k value
    for k in k_values:
        quantized_image = kmeans_quantization(original_image, k)
        mse = calculate_mse(original_image, quantized_image)
        mse_values.append(mse)

        # Display the original and quantized images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(quantized_image)
        plt.title(f'Quantized Image (k={k}) - MSE: {mse:.2f}')
        plt.axis('off')

        plt.show()

    # Analysis of the effectiveness and limitations of k-Means image quantization
    print("Effectiveness and Limitations Analysis:")
    for i, k in enumerate(k_values):
        print(f"k={k}, MSE={mse_values[i]:.2f}")

    print("\nDiscussion:")
    print("As the value of k increases, the MSE between the original and quantized images generally decreases,")
    print("indicating that the quantized image becomes more similar to the original image.")
    print("However, increasing k also increases the computational cost and may lead to diminishing returns.")
    print("In cases with too low a k value, the image may lose significant detail, resulting in a higher MSE.")
    print("Thus, k-Means image quantization is effective at reducing the number of colors in an image, but")
    print("the choice of k is crucial in balancing image quality and computational efficiency.")


if __name__ == "__main__":
    main()
