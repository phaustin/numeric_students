#!/usr/bin/env python
"""
This module provides the plot_secant function. 

Example usage from python: ::

     >> import lab1_secant

     >> lab1_secant.plot_secant(2)

This should produce a figure window with a simple plot

"""

import numpy as np
import matplotlib.pyplot as plt

def plot_secant(xb):
    """This function draws a secant from x=1 to x=xb to the curve x**3 - 5*x.
    Plots the curve and the secant."""

    # Define curve
    x = np.arange(-3., 3.2, .2)
    y = x**3 - 5*x

    # Find secant points
    xs = np.array([1, xb])
    ys = xs**3 - 5*xs

    # find straight line between secant points
    m = (ys[1] - ys[0]) / (xs[1] - xs[0])
    b = ys[0] - m*xs[0]

    # Extend secant line beyond points
    xc = np.array( [xs[0] - 0.5, xs[1] + 0.5] )
    yc = m*xc + b

    # Plot the lines and points
    plt.plot(x, y)
    plt.plot(xs, ys, 'o')
    plt.plot(xc, yc)
    plt.show()

if __name__ == '__main__':
    # Test command
    plot_secant(2.)
