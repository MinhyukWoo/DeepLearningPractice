"""
Notice: This project is not commercial. Just for educational.
Indirectly and directly quoted from numpy.org
https://numpy.org/doc/stable/user/absolute_beginners.html#how-to-import-numpy
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_division_line():
    print("="*30)


if __name__ == '__main__':
    """
    The datatype ndarray is shorthand for "N-dimensional array".
    Vector is an array with a single dimension.
    Matrix is an array with two dimension.
    Tensor is an array with 3-D or higher dimension.
    In NumPy, dimensions are called axes.
    
    In NumPy, most of copy operator is a shallow copy.
    It means modifying data also modifies the original array.
    If you want a deep copy, use .copy() 
    """
    # Create a basic array
    basic_vector1 = np.array([4, 1, 2, 3, 9, 5])
    print(basic_vector1)
    print(np.empty(3))
    print(np.zeros(3))
    print(np.ones(4))
    print(np.arange(5))
    print(np.arange(-1, 11, 2))
    print(np.linspace(0, 10, num=5))
    print(np.ones(2, dtype=np.int64))
    # Adding, removing, and sorting elements
    print_division_line()
    print(np.sort(basic_vector1))
    basic_vector2 = np.array([1, 3, 4])
    print(np.concatenate((basic_vector1, basic_vector2)))
    basic_matrix1 = np.array([[1, 2], [5, 6], [3, 0]])
    basic_matrix2 = np.array([[3, 4], [0, 2], [7, 9]])
    print(np.concatenate((basic_matrix1, basic_matrix2), axis=0))
    # Get the shape and size of an array
    print_division_line()
    print(basic_matrix1)
    print("ndim:", basic_matrix1.ndim)
    print("size:", basic_matrix1.size)
    print("shape:", basic_matrix1.shape)  # datatype is tuple
    # Reshape an array
    print_division_line()
    print(basic_matrix1)
    print("---reshape-->")
    reshaped1 = basic_matrix1.reshape(2, 3)  # newshape=(2, 4) order='C' C-like
    print(reshaped1)
    print(np.reshape(basic_matrix1, newshape=(1, 6), order='F'))  # F-like
    # Add a new axis to an array
    print_division_line()
    print(basic_vector1)
    print("row_vector")
    row_vector = basic_vector1[np.newaxis, :]
    print(row_vector)
    print("col_vector")
    col_vector = basic_vector1[:, np.newaxis]
    print(col_vector)
    converted = basic_vector1[np.newaxis, np.newaxis, :]
    converted = np.reshape(converted, (2, 3, 1))
    print(converted)
    # Indexing and slicing
    print_division_line()
    print(basic_matrix1[0][:] == basic_matrix1[0, :])
    print(basic_vector1)
    print(basic_vector1[basic_vector1 % 2 == 1])
    print(basic_vector1 % 2 == 1)
    print((basic_vector1 % 2 == 1) & (basic_vector1 < 7))
    print(np.nonzero(basic_vector1 > 3))  # print the indices of elements
    print(basic_vector1[np.nonzero(basic_vector1 > 3)])
    # Create an array from existing data
    print_division_line()
    print("np.vstack")
    print(np.vstack((basic_matrix1, basic_matrix2)))
    print("np.hstack")
    print(np.hstack((basic_matrix1, basic_matrix2)))
    print(basic_matrix1[0, 0])
    # Basic array operations
    print_division_line()
    print("[2, 3, 5] And [1, -2, 3]")
    print(np.array([2, 3, 5] + np.array([1, -2, 3])))
    print(np.array([2, 3, 5] * np.array([1, -2, 3])))
    print(np.sum(np.arange(1, 11)))
    print(basic_matrix1)
    print(np.sum(basic_matrix1, axis=0))
    print(np.ones(3) * 2)
    print(basic_matrix1.max(), basic_matrix1.min(), basic_matrix1.sum())
    # Generating random numbers
    print_division_line()
    rng = np.random.default_rng(int(time.time()))
    print(rng.random(3))
    print(rng.integers(5, size=(2, 4)))
    # Get unique items and counts
    print_division_line()
    basic_vector3 = np.array([11, 11, 12, 13, 14,
                              15, 16, 11, 12, 13])
    print(basic_vector3)
    print(np.unique(basic_vector3))
    print(np.unique(basic_vector3, return_index=True))
    print(np.unique(basic_vector3, return_counts=True))
    # Transpose a matrix
    print(basic_matrix1)
    print(basic_matrix1.transpose())
    print(basic_matrix1.T)
    # reverse an array
    print("flip")
    print(np.flip(np.arange(1, 11)))
    # flatten multidimensional arrays
    print("flatten")
    print(basic_matrix1.flatten())  # deep copy
    print(basic_matrix1.ravel())  # shallow copy, using view
    # save and load NumPy objects
    # save : save a single object to a binary file(.npy)
    # savez : save multiple objects to a binary file(.npz)
    # savetxt : save a single object to a text file(.txt or .csv)
    # import and export a CSV with pandas library
    # import pandas as pd
    # plot arrays with matplotlib library
    # import matplotlib.pyplot as plt
    plt.plot(basic_vector1)
    plt.show()

