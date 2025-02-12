from PIL import Image, ImageOps
import math

# Open the image
img = Image.open(
    r"C:\Users\mites\OneDrive\Desktop\13549827_web1_180914-CDT-booknotes.jpg")

# Convert image to grayscale if it's not already
if img.mode != 'L':
    img = img.convert('L')

# Get image dimensions
width, height = img.size

# Create a new image for the result
result = Image.new('L', (width, height))

# Define the constant for scaling
c = 255 / math.log(256)  # this scales the result to fit 8-bit range [0, 255]

# Process each pixel
for x in range(width):
    for y in range(height):
        r = img.getpixel((x, y))
        # Apply logarithmic transformation
        s = int(c * math.log(1 + r))
        # Ensure the value is within 0-255 range
        s = max(0, min(s, 255))
        result.putpixel((x, y), s)

# Save the result image
result.save("log_transformed_Lena.png")
