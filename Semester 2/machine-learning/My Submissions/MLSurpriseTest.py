import numpy as np

# Conditional Sampling function


def conditional_sampling(current_x, sampling_index, condition_index, mean, cov):
    # Extract a, b, c from the covariance matrix
    a = cov[sampling_index, sampling_index]
    b = cov[sampling_index, condition_index]
    c = cov[condition_index, condition_index]

    # Compute conditional mean and standard deviation (sigma)
    mu = mean[sampling_index] + \
        (b * (current_x[condition_index] - mean[condition_index]) / c)
    sigma = np.sqrt(a - (b ** 2) / c)

    # Sample new x for the sampling index
    new_x = np.copy(current_x)
    new_x[sampling_index] = np.random.randn() * sigma + mu

    return new_x

# Main Gibbs sampling function


def gibbs_sampling(initial_points, cov, num_samples):
    # Initialize the sampler to store the samples
    sampler = np.empty((num_samples + 1, 2))

    # Set the initial points (x0, x1)
    sampler[0] = initial_points
    x0, x1 = initial_points

    # Run the sampling loop
    for i in range(num_samples):
        # Sample x0 given x1
        current_x = np.array([x0, x1])
        x0 = conditional_sampling(
            current_x, sampling_index=0, condition_index=1, mean=[0, 0], cov=cov)[0]

        # Sample x1 given x0
        current_x = np.array([x0, x1])
        x1 = conditional_sampling(
            current_x, sampling_index=1, condition_index=0, mean=[0, 0], cov=cov)[1]

        # Store the new sample
        sampler[i + 1] = [x0, x1]

    return sampler


# Initial point, covariance matrix, and number of samples
initial_points = [2, 3]
cov = np.array([[10, 3], [3, 5]])
num_samples = 10

# Perform Gibbs sampling
samples = gibbs_sampling(initial_points, cov, num_samples)

# Output the samples
print(samples)
