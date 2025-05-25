from PIL import Image
import os

INPUT_DIR = "C:\\mtechpracticalsdatasets\\janu\\dataset"
OUTPUT_DIR = "C:\\mtechpracticalsdatasets\\janu\\512x512"
TARGET_SIZE = 512  # or 256, 1024 based on your GPU

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(INPUT_DIR, filename)
        img = Image.open(img_path).convert("RGB")

        # Make it square by padding
        width, height = img.size
        size = max(width, height)
        square_img = Image.new("RGB", (size, size), (0, 0, 0))
        square_img.paste(img, ((size - width) // 2, (size - height) // 2))

        # Resize to target
        resized_img = square_img.resize((TARGET_SIZE, TARGET_SIZE))
        resized_img.save(os.path.join(OUTPUT_DIR, filename))
