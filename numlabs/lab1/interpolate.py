#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

def interpol_f(pn):   

     """
     Uses three different methods to interpolate the function :math:`f(x)=x^{3} - 5x`

     Parameters
     ----------
     pn : integer
        number of points  in f(x)

     Returns
     -------
     (x,y) : (array,array)
        tupple (x(pn) , y(pn)) with analytic values y=f(x)
        in the range :math:`-5 < x < 5`

        plot is produced as a side effect, with three lines of x=101 points
        showing the analytic function, linear interpolation and
        cubic spline interpolation
     """

     npts = 101
     ddx = 10./(npts-1)
     dx = 10./(pn-1)
     x = np.zeros((pn,))
     xx = np.zeros((npts,))
     y = np.zeros((pn,))
     act = np.zeros((npts,))

     for i in range(pn):
         x[i] = -5 + i*dx
         y[i] = x[i]**3 - 5 * x[i]

     for i in range(npts):
         xx[i] = -5 + i*ddx
         act[i] = xx[i]**3 - 5 * xx[i]

     interpolater = interp1d(x, y, kind='cubic')
     zz = interpolater(xx)

     plt.plot(x, y, 'bo')
     plt.title('Linear blue, Cubic Black, Actual Red')
     plt.plot(x,y,'b')
     plt.plot(xx, act, 'r')
     plt.plot(xx, zz, 'k')

     plt.show()

     return (x,y)

def interpol_g(pn):   

     """
     Uses three different methods to interpolate the function :math:`f(x)=|x|^{2/3}`

     Parameters
     ----------
     pn : integer
        number of points in f(x)

     Returns
     -------
     (x,y) : (array,array)
        tupple (x(pn) , y(pn)) with analytic values y=f(x)
        in the range :math:`-5 < x < 5`

        plot is produced as a side effect, with three lines of x=101 points
        showing the analytic function, linear interpolation and
        cubic spline interpolation
     """

     npts = 101
     ddx = 10./(npts-1)
     dx = 10./(pn-1)
     x = np.zeros((pn,))
     xx = np.zeros((npts,))
     y = np.zeros((pn,))
     act = np.zeros((npts,))

     for i in range(pn):
         x[i] = -5 + i*dx
         y[i] = np.abs(x[i])**(2./3.)

     for i in range(npts):
         xx[i] = -5 + i*ddx
         act[i] = np.abs(xx[i])**(2./3.)

     interpolater = interp1d(x, y, kind='cubic')
     zz = interpolater(xx)

     plt.plot(x, y, 'bo')
     plt.title('Linear blue, Cubic Black, Actual Red')
     plt.plot(x,y,'b')
     plt.plot(xx, act, 'r')
     plt.plot(xx, zz, 'k')

     plt.show()

     return (x,y)

if __name__ == '__main__':
     # Test command
     interpol(6)
