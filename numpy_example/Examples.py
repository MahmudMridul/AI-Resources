import numpy as np

# initialize an array
a = np.array([1, 2, 3, 4, 5])
a2D = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# gets dimension
dim = a2D.ndim

# gets number of rows and columns
shape = a2D.shape

# gets number of bytes one item takes in memory
item_size = a2D.itemsize

# gets total number of bytes array takes in memory
total_mem = a2D.nbytes

# access specific element
a2D[2, 0] = 100
elem = a2D[2, 0]

# gets a row
row = a2D[2, :]

# gets a column
a2D[:, 2] = np.array([10, 12, 16])
col = a2D[:, 2]

# gets elements in specified range [start_index : end_index : step]
range_elem = a2D[1:3, 1:3:2]

# creates array with specified shape (5, 3) having random numbers from 0 to 1
random_1 = np.random.rand(5, 3)

# same as previous but can take tuple as parameter
random_2 = np.random.random_sample((3, 3))

# get data from text file
numbers = np.genfromtxt('numbers.txt', delimiter=',', dtype='int32')

# returns an array of same size with boolean values
nums_greater_five = numbers > 5

# pass expression to get values
nums_smaller_five = numbers[numbers < 5]

# pass list as index
specific_indexes = a[[1, 2, 4]]

# -1 refers to automatic or length of given array
c = np.array([5, 7, 2])
c_reshape = np.reshape(c, (-1, 1))

# horizontally stack two arrays
h_stacked = np.hstack((a2D, c_reshape))




