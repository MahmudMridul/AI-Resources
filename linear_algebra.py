import numpy as np

# here v and w represents 2D vectors
# v = 2i + 3j [i and j represent unit vectors i hat and j hat]
# w = -3i + 5j [i and j represent unit vectors i hat and j hat]
v = np.array([2, 3])
w = np.array([-3, 5])

# a linear transformation of a 2D vector can be
# represented with a 2*2 matrix
# this matrix represent the coordinates where i and j
# landed after transformation
# here i = [0, 1] j = [-1, 0]
# this is a 90-degree right rotation
t = np.array([[0, -1], [1, 0]])

# to know the coordinates of v after
# 90-degree right rotation we need to
# multiply t*v
v_transformed = np.dot(t, v)
print(v_transformed)
