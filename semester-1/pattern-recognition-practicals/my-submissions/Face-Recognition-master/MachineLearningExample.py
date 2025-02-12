import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a simple dataset
# Let's assume the relationship between house size (in square feet) and price (in $1000s) is linear.
np.random.seed(42)
house_size = 2 * np.random.rand(100, 1)  # House sizes between 0 and 2 (e.g., 1000 to 2000 sq ft)
house_price = 4 + 3 * house_size + np.random.randn(100, 1)  # Price = 4 + 3*size + noise

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(house_size, house_price, test_size=0.2, random_state=42)

# Step 3: Train the machine learning model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions using the trained model
y_pred = model.predict(X_test)

# Step 5: Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Step 6: Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')
plt.xlabel('House Size (1000s of sq ft)')
plt.ylabel('House Price ($1000s)')
plt.legend()
plt.show()
