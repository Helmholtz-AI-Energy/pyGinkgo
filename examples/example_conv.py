# SPDX-FileCopyrightText: 2025 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

# importing necessary modules
import numpy as np
import sys

sys.path.append("/Users/go/Code/pyGinkgo/build/Develop")

import pyGinkgo as pg

ref = pg.device("cuda")
values_1 = [[3.0, 2.0, -1.0], [-1.0, 5.0, 6.0], [4.0, 4.0, 9.0]]
np_arr_1 = np.array(values_1)
n_rows, n_columns = np_arr_1.shape
dense_1 = pg.matrix.dense_double(ref, np.array(values_1))

values_2 = [[3.0, 2.0, -1.0], [-1.0, 5.0, 6.0], [4.0, 3.0, 9.0]]
np_arr_2 = np.array(values_2)
n_rows, n_columns = np_arr_2.shape
dense_2 = pg.matrix.dense_double(ref, np.array(values_2))

conv = pg.matrix.conv2d_double(ref, [dense_1, dense_2])


# input image
values_input = [
    [3.0, 2.0, -1.0, 4.0, 5.0],
    [-1.0, 5.0, 6.0, 4.0, 5.0],
    [4.0, 4.0, 9.0, 4.0, 5.0],
    [3.0, 5.0, 6.0, 4.0, 5.0],
    [1.0, 2.0, 3.0, 4.0, 5.0],
]
np_arr_input = np.array(values_input)
n_rows_input, n_columns_input = np_arr_input.shape
input_image = pg.matrix.dense_double(ref, np.array(values_input))

padding_row = 0
padding_col = 0
stride_row = 1
stride_col = 1

output_size_row = (n_rows_input + 2 * padding_row - n_rows) // stride_row + 1
output_size_col = (n_columns_input + 2 * padding_col - n_columns) // stride_col + 1


# Create Dense<double> of the correct shape
output_1 = pg.matrix.dense_double(ref, (output_size_row, output_size_col))

# Fill with zeros
output_1.fill(0.0)

# Create Dense<double> of the correct shape
output_2 = pg.matrix.dense_double(ref, (output_size_row, output_size_col))

# Fill with zeros
output_2.fill(0.0)


# Apply convolution with multiple output kernels
conv.apply_multi(input_image, [output_1, output_2])

# Wait for device computations to complete
ref.synchronize()
