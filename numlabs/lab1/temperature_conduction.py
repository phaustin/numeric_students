#!/usr/bin/env python
"""
Script to plot the exponential decay of temperature of a number of objects
with different initial temperature in a single ambient temperature.

Example usage, use default values:

>> lab1_temperature()

Example usage, set Ta=10, To=(-5, 5, 10, 15) and la (lambda)= 0.0002:

>> lab1_temperature(10, array([-5., 5., 10., 15.]), la = 0.0002)
     
Ta is the ambient temperature (Celcius)
To is the initial temperature (four different cases) (Celcius)
la is lambda, the time constant of equilibriations (1/second

"""

import matplotlib.pyplot as plt
import numpy as np

def temperature(Ta = 20, To = np.array([-10., 10., 20., 30.]), la = 0.00001):

    # set the time scale (seconds)
    t = np.arange(0., 400000., 100.)

    # calculate the temperatures with time of the four objects
    T = np.zeros((To.shape[0], t.shape[0]))
    for i, Tinitial in enumerate(To):
        T[i,:] = Ta + (Tinitial - Ta) * np.exp(-la * t)

    # plot the temperatures in hours
    t=t/3600.
    for i, Tinitial in enumerate(To):
        plt.plot(t, T[i], label="To = %s" % Tinitial)
        
    # label axes    
    plt.xlabel('time (hours)')
    plt.ylabel('temperature (deg C)')

    # add a legend
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    # Test command
    temperature()
