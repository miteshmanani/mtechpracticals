import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load an image and convert to grayscale
image = cv2.imread('C:\\Users\\mites\\OneDrive\\Desktop\\The-famous-Lena-image-often-used-as-an-example-in-image-processing.png', cv2.IMREAD_GRAYSCALE)

# Ensure the image dimensions are divisible by 4 for simplicity
height, width = image.shape
height -= height % 4
width -= width % 4
image = image[:height, :width]

# Block size
block_size = 4

# Calculate number of blocks
num_blocks_vertical = height // block_size
num_blocks_horizontal = width // block_size

# Create subplots dynamically
fig, axes = plt.subplots(num_blocks_vertical, num_blocks_horizontal, figsize=(15, 15))

# Loop through each block and display it
for i in range(num_blocks_vertical):
    for j in range(num_blocks_horizontal):
        # Extract the block
        block = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
        
        # Display the block in its subplot
        ax = axes[i, j]
        ax.imshow(block, cmap='gray')
        ax.axis('off')

# Adjust layout and show
plt.tight_layout()
plt.show()
