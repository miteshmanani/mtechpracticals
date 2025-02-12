import numpy as np

# Given dataset
data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

# Step 1: Center the data
mean = np.mean(data, axis=0)
centered_data = data - mean

# Step 2: Compute the covariance matrix
cov_matrix = np.cov(centered_data, rowvar=False)

# Step 3: Find the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# First principal component
first_pc = sorted_eigenvectors[:, 0]

# Step 5: Project the centered data onto the first principal component
projected_data = centered_data.dot(first_pc)

projected_data, sorted_eigenvalues, sorted_eigenvectors

print("Projected Data : {} ".format(projected_data))
print("Sorted Eigen Values : {} ".format(sorted_eigenvalues))
print("Sorted Eigen Vectors : {} ".format(sorted_eigenvectors))
