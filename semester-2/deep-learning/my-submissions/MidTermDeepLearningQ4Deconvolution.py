import numpy as np

output_matrix = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

filter_matrix = np.ones((3, 3))
stride = 2

output_height = (output_matrix.shape[0] - 1) * stride + filter_matrix.shape[0]
output_width = (output_matrix.shape[1] - 1) * stride + filter_matrix.shape[1]

deconv_output = np.zeros((output_height, output_width))

for i in range(output_matrix.shape[0]):
    for j in range(output_matrix.shape[1]):
        deconv_output[i * stride:i * stride + filter_matrix.shape[0],
                      j * stride:j * stride + filter_matrix.shape[1]] += output_matrix[i, j] * filter_matrix

print("Padding-Free Deconvolution Result:")
print(deconv_output)

# ---- Deconvolution with Zero Padding ----
# Adding a zero padding of 1 around the 3x3 output matrix for deconvolution with zero padding
padded_output_matrix = np.pad(
    output_matrix, pad_width=1, mode='constant', constant_values=0)

# Calculating the dimensions of the resulting deconvolved matrix with zero-padding
padded_output_height = (
    padded_output_matrix.shape[0] - 1) * stride + filter_matrix.shape[0]
padded_output_width = (
    padded_output_matrix.shape[1] - 1) * stride + filter_matrix.shape[1]

# Initialize deconvolution output with zeros (with zero-padding)
deconv_output_padded = np.zeros((padded_output_height, padded_output_width))

# Perform the deconvolution (transposed convolution) operation with zero-padded output
for i in range(padded_output_matrix.shape[0]):
    for j in range(padded_output_matrix.shape[1]):
        deconv_output_padded[i * stride:i * stride + filter_matrix.shape[0],
                             j * stride:j * stride + filter_matrix.shape[1]] += padded_output_matrix[i, j] * filter_matrix

print("\nDeconvolution with Zero Padding Result:")
print(deconv_output_padded)
