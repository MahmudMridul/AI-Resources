from scipy.sparse import csr_matrix
import numpy as np

divider = "========== ========== ========== ========== =========="

mat_1 = np.array([[1, 2, 3], [4, 3, 2]])
mat_2 = np.array([[1, 0, 0], [0, 3, 0]])

mat_1_sparsed = csr_matrix(mat_1)
mat_2_sparsed = csr_matrix(mat_2)

# print(mat_1_sparsed)
# print(divider)
# print(mat_2_sparsed)
"""
csr_matrix stores only non zeros
using csr_matrix is useful when there are many 0's in data. 
"""
