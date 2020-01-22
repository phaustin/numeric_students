#!/usr/bin/env python
"""
Sample script that will generate 2 matrices, and graph them.

Two (of many) different methods of populating n-dimensional arrays are
shown.

Example usage from Python::

  >>> import lab3_sample

  >>> lab3_sample.sample()

This should print the 2 matrices, and produce a figure window with a
simple plot of the matrices as lines in 3-space.

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sample():
    """Generate 2 matrices using different techniques to populate
    them, and graph them.
    """
    # Generate the 1st matrix.  Start with 10-row, 3-column array of
    # zeros, and assign element values.
    #
    # Note that (in contrast to Matlab) Python arrays use 0-based
    # indexing, and that the NumPy method zeros creates an array of
    # floats by default.
    matrix1 = np.zeros((10, 3))
    for i in xrange(10):
        matrix1[i, 0] = i + 1
        matrix1[i, 1] = i + 2
        matrix1[i, 2] = i + 3

    # Generate the 2nd matrix by combining the results of calls of
    # the NumPy arange method.
    #
    # Note that NumPy arrays are created row-wise, so we transpose to
    # get the shape we want.
    matrix2 = np.array((np.arange(-1., -31., -3.),
                        np.arange(0., -30., -3.),
                        np.arange(1., -29., -3.))).transpose()

    # Display the matrices
    print 'matrix1:\n', matrix1
    print 'matrix2:\n', matrix2

    # Graph the matrices as lines in 3-space
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(xs=matrix1[:,0], ys=matrix1[:,1], zs=matrix1[:,2],
            color='b', label='matrix1')
    ax.plot(xs=matrix2[:,0], ys=matrix2[:,1], zs=matrix2[:,2],
            color='r', label='matrix2')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # Test command
    sample()
