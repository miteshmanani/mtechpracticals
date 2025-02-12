from PIL import Image, ImageOps

# Open the image
img = Image.open(
    r"C:\Users\mites\OneDrive\Desktop\The-famous-Lena-image-often-used-as-an-example-in-image-processing.png")

# Convert image to grayscale if it's not already
if img.mode != 'L':
    img = img.convert('L')

# Get image dimensions
width, height = img.size

# Create a new image for the result
result = Image.new('L', (width, height))

# Hardcode the gamma value here (you can change this to see different effects)
gamma = 1  # Example: 0.5 for darker, 2.0 for lighter

# Define the constant for scaling
c = 255 ** (1 - gamma)  # This ensures the output fits in [0, 255]

# Process each pixel
for x in range(width):
    for y in range(height):
        r = img.getpixel((x, y))
        # Apply power-law (gamma) transformation
        s = int(c * r ** gamma)
        # Ensure the value is within 0-255 range
        s = max(0, min(s, 255))
        result.putpixel((x, y), s)

# Save the result image
result.save("gamma_transformed_Lena.png")
