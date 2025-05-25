from rembg import remove
from PIL import Image
import os

# Directory containing images
# Replace with your directory path
input_dir = r"C:\mtechpracticalsdatasets\janu\dataset\class_1"
# Replace with your output directory path
output_dir = r"C:\mtechpracticalsdatasets\janu\dataset"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Supported image extensions
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Process each image in the directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(
            output_dir, f"no_bg_{filename.split('.')[0]}.png")

        # Open the image
        with open(input_path, "rb") as input_file:
            img_data = input_file.read()

        # Remove the background
        output_data = remove(img_data)

        # Save the result as PNG to preserve transparency
        with open(output_path, "wb") as output_file:
            output_file.write(output_data)

        print(f"Processed: {filename} -> Saved as: {output_path}")

print("All images processed!")
