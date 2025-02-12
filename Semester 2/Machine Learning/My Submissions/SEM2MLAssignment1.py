import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to generate the dataset


def generate_data(n_samples):
    np.random.seed(42)
    x = np.linspace(0, 1, n_samples).reshape(-1, 1)
    y = np.sin(1 + x**2) + np.random.normal(0, 0.03, n_samples).reshape(-1, 1)
    return x, y

# Function to perform polynomial regression and calculate MSE


def polynomial_regression_analysis(x_train, y_train, x_test, y_test, degrees):
    train_errors = []
    test_errors = []

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        x_poly_train = poly_features.fit_transform(x_train)
        x_poly_test = poly_features.transform(x_test)

        model = LinearRegression()
        model.fit(x_poly_train, y_train)

        y_train_pred = model.predict(x_poly_train)
        y_test_pred = model.predict(x_poly_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

    return train_errors, test_errors

# Function to plot MSE for training and testing sets


def plot_mse(degrees, train_errors, test_errors, title):
    plt.plot(degrees, train_errors, label='Training MSE', marker='o')
    plt.plot(degrees, test_errors, label='Testing MSE', marker='o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(title)
    plt.legend()
    plt.show()

# Function to plot polynomial fits


def plot_polynomial_fits(x, y, degrees, x_range, n_samples):
    plt.scatter(x, y, color='blue', label='Data Points', s=10)

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(x_range)
        model = LinearRegression()
        model.fit(poly_features.fit_transform(x), y)
        y_poly_pred = model.predict(x_poly)
        plt.plot(x_range, y_poly_pred, label=f'Degree {degree}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Polynomial Fits (n={n_samples})')
    plt.legend()
    plt.show()


# Parameters
degrees = [0, 1, 2, 3, 4, 5, 6]
x_range = np.linspace(0, 1, 500).reshape(-1, 1)

# Analysis for 50 data points
x, y = generate_data(50)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
train_errors, test_errors = polynomial_regression_analysis(
    x_train, y_train, x_test, y_test, degrees)
plot_mse(degrees, train_errors, test_errors,
         title='MSE for Polynomial Degrees (n=50)')
plot_polynomial_fits(x_train, y_train, degrees, x_range, 50)

# Analysis for 1000 data points
x, y = generate_data(1000)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
train_errors, test_errors = polynomial_regression_analysis(
    x_train, y_train, x_test, y_test, degrees)
plot_mse(degrees, train_errors, test_errors,
         title='MSE for Polynomial Degrees (n=1000)')
plot_polynomial_fits(x_train, y_train, degrees, x_range, 1000)
