import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Generate initial dataset
x_initial = np.linspace(0, 1, 50)
y_initial = np.sin(1 + x_initial**2) + \
    np.random.normal(0, 0.03, len(x_initial))

# Step 2: Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_initial, y_initial, test_size=0.2, random_state=42)

# Step 3: Fit polynomial regression models and calculate MSE
degrees = range(7)
train_mse = []
test_mse = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    x_poly_train = poly.fit_transform(x_train.reshape(-1, 1))
    x_poly_test = poly.transform(x_test.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_poly_train, y_train)

    y_train_pred = model.predict(x_poly_train)
    y_test_pred = model.predict(x_poly_test)

    train_mse.append(mean_squared_error(y_train, y_train_pred))
    test_mse.append(mean_squared_error(y_test, y_test_pred))

# Step 4: Visualize the fits
plt.figure(figsize=(10, 6))
plt.plot(x_initial, y_initial, 'o', label='Data')
for degree in degrees:
    plt.plot(x_initial, model.predict(poly.transform(
        x_initial.reshape(-1, 1))), label=f'Degree {degree}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Fits')
plt.legend()
plt.show()

# Step 5: Repeat with more data points
x_large = np.linspace(0, 1, 1000)
y_large = np.sin(1 + x_large**2) + np.random.normal(0, 0.03, len(x_large))

# Step 6: Fit models and calculate MSE for larger dataset
train_mse_large = []
test_mse_large = []

for degree in degrees:
    poly_large = PolynomialFeatures(degree=degree)
    x_poly_large = poly_large.fit_transform(x_large.reshape(-1, 1))

    model_large = LinearRegression()
    model_large.fit(x_poly_large, y_large)

    y_large_pred = model_large.predict(x_poly_large)

    train_mse_large.append(mean_squared_error(y_large, y_large_pred))

# Step 7: Compare results
print("MSE for initial dataset (degrees 0 to 6):", test_mse)
print("MSE for larger dataset (degrees 0 to 6):", train_mse_large)
