import numpy as np
import matplotlib.pyplot as plt

# Number of data points
n = 50

# Generate x values uniformly between 0 and 1
x = np.random.uniform(0, 1, n)

# Calculate y values using the given formula
y = np.sin(1 + x**2)

# Add noise with normal distribution N(0, 0.03^2)
noise = np.random.normal(0, 0.03, n)
y_noisy = y + noise

# Plot the data
plt.scatter(x, y_noisy, label='Noisy Data')
plt.plot(x, y, color='red', label='Original Data (without noise)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
