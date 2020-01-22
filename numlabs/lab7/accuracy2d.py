#!/usr/bin/env python
"""Plot an actual dispersion relation, and versions of it discretized
on 2 different grids to illustrate the accuracy of different
discretization schemes.

This is an illustration of lab7 section 6.1.

Example usage from ipython::

  $ ipython -pylab
  ...
  In [1]: run accuracy2d
  # Run with doR = 0.5
  In [2]: run accuracy2d 0.5
"""
import sys
import numpy as np
import matplotlib.pyplot as plt


def main(arg):
    """Calculate and plot the dispersion relations.

    arg is doR, the ratio of the grid size, d, to the Rossby radius, R.
    """
    doR = float(arg)
    # Domain to plot over
    kd = np.linspace(-np.pi, np.pi)
    # Actual dispersion relation
    actual = 1 + kd ** 2 / doR ** 2
    # Dispersion relation discretized on grid 1
    grid1 = np.cos(kd / 2) ** 2 + 4 * np.sin(kd / 2) ** 2 / doR ** 2
    # Dispersion relation discretized on grid 2
    grid2 = 1 + 4 * np.sin(kd / 2) ** 2 / doR ** 2
    # Plot the actual and discretized dispersion relations
    plt.plot(kd, actual, kd, grid1, kd, grid2)
    plt.xlim((-np.pi, np.pi))
    plt.legend(('Actual', 'Grid 1', 'Grid 2'), loc='upper center')
    plt.xlabel('$k d$')
    plt.ylabel('$\omega / f$')
    plt.title('$d / R = %.3f$' % doR)
    plt.show()


if __name__ == '__main__':
    # sys.argv is the command-line arguments as a list. It includes
    # the script name as its 0th element. Check for the degenerate
    # cases of no aditional arguments, or the 0th element containing
    # `sphinx-build`. The latter is a necessary hack to accommodate
    # the sphinx plot_directive extension that allows this module to
    # be run to include its graph in sphinx-generated docs.
    if len(sys.argv) == 1 or 'sphinx-build' in sys.argv[0]:
        # Default to doR = 1
        main(1)
    elif len(sys.argv) == 2:
        # Run with the value of doR the user gave
        main(sys.argv[1])
    else:
        print ('Usage: accuracy2d doR')
        print ('where doR is the ratio of grid size, d, to Rossby radius, R')
