import os
from PIL import Image


def bit_plane_slice(image_path, i):
    # Open the image
    # Convert to grayscale if not already
    img = Image.open(image_path).convert('L')
    width, height = img.size

    # Create a new image for the result
    result = Image.new('L', (width, height))

    # Process each pixel
    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))
            # Extract the i-th bit from the pixel value
            bit = (pixel >> i) & 1
            # Set the pixel in the result image; multiply by 255 to make it visible
            result.putpixel((x, y), bit * 255)

    # Get just the filename without path
    filename = os.path.basename(image_path)
    # Remove the extension
    filename_without_ext = os.path.splitext(filename)[0]

    # Save the result image with appended bit plane number
    save_path = os.path.join(os.path.dirname(image_path), f"bit_plane_{
                             i}_{filename_without_ext}.png")
    result.save(save_path)


# Example usage:
image_path = r"C:\Users\mites\OneDrive\Desktop\13549827_web1_180914-CDT-booknotes.jpg"
for i in range(8):  # 8 bit planes for an 8-bit image
    bit_plane_slice(image_path, i)
