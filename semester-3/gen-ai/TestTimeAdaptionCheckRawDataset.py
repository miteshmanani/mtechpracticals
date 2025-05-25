import struct
import numpy as np

# Read training images
with open('./data/MNIST/raw/train-images-idx3-ubyte', 'rb') as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

# Read training labels
with open('./data/MNIST/raw/train-labels-idx1-ubyte', 'rb') as f:
    magic, num = struct.unpack(">II", f.read(8))
    labels = np.frombuffer(f.read(), dtype=np.uint8)

# Display an example
import matplotlib.pyplot as plt

plt.imshow(images[0], cmap='gray')
plt.title(f"Label: {labels[0]}")
plt.show()
