from PIL import Image

# Open the image
img = Image.open(
    r"C:\Users\mites\OneDrive\Desktop\The-famous-Lena-image-often-used-as-an-example-in-image-processing.png")

# Convert image to RGB mode if it's not already
if img.mode != 'RGB':
    img = img.convert('RGB')

# Get image dimensions
width, height = img.size

# Create a new image for the result
result = Image.new('RGB', (width, height))

# Process each pixel
for x in range(width):
    for y in range(height):
        r, g, b = img.getpixel((x, y))

        # Apply negative transformation for each channel
        # L = 256 for 8-bit images
        new_r = 255 - r
        new_g = 255 - g
        new_b = 255 - b

        # Set the new pixel
        result.putpixel((x, y), (new_r, new_g, new_b))

# Save the result image
result.save("negative_Lena.png")
