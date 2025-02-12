import numpy as np

# Initial matrix (5x5 as in the image)
input_matrix = np.array([
    [10, 12, 11, 13, 14],
    [11, 12, 11, 87, 23],
    [23, 32, 79, 86, 8],
    [65, 78, 8, 68, 26],
    [67, 97, 92, 21, 15]
])

# Padding the matrix with zeros (padding = 1)
input_padded = np.pad(input_matrix, pad_width=1,
                      mode='constant', constant_values=0)

# Filter (3x3) - Using a simple averaging filter for demonstration
filter_matrix = np.ones((3, 3))

# Parameters
stride = 2
bias = 6

# Sigmoid activation function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to perform the convolution operation


def convolution_with_stride(input_matrix, filter_matrix, stride, bias):
    # Dimensions of the output matrix
    output_height = (input_matrix.shape[0] -
                     filter_matrix.shape[0]) // stride + 1
    output_width = (input_matrix.shape[1] -
                    filter_matrix.shape[1]) // stride + 1

    # Initialize the output matrix
    output_matrix = np.zeros((output_height, output_width))

    # Convolution operation
    for i in range(0, input_matrix.shape[0] - 2, stride):
        for j in range(0, input_matrix.shape[1] - 2, stride):
            # Extract the 3x3 sub-matrix
            sub_matrix = input_matrix[i:i+3, j:j+3]
            # Convolution (sum of element-wise multiplication)
            conv_result = np.sum(sub_matrix * filter_matrix)
            # Add bias and apply sigmoid
            activated_result = sigmoid(conv_result + bias)
            # print(activated_result)
            # Calculate the position in the output matrix
            output_i, output_j = i // stride, j // stride
            output_matrix[output_i, output_j] = activated_result

    return output_matrix


# Perform the convolution
output_matrix = convolution_with_stride(
    input_padded, filter_matrix, stride, bias)

# Display the input, padded, and output matrices
print("Input Matrix:")
print(input_matrix)
print("\nPadded Input Matrix:")
print(input_padded)
print("\nOutput Matrix after Convolution, Bias Addition, and Sigmoid Activation:")
print(output_matrix)
