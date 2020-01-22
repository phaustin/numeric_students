"""
  this script show how to plot the heat conduction equation

"""
import matplotlib.pyplot as plt
import context
from numlabs.lab2.lab2_functions import euler,beuler,leapfrog
import numpy as np


theFuncs={'euler':euler,'beuler':beuler,'leapfrog':leapfrog}

if __name__=="__main__":
    tend=10.
    Ta=20.
    To=30.
    theLambda=-8.
    funChoice='euler'
    npts=40
    approxTime,approxTemp=theFuncs[funChoice](npts,tend,To,Ta,theLambda)
    exactTime=np.empty([npts,],float)
    exactTemp=np.empty_like(exactTime)
    for i in np.arange(0,npts):
       exactTime[i] = tend*i/npts
       exactTemp[i] = Ta + (To-Ta)*np.exp(theLambda*exactTime[i])
    plt.close('all')
    plt.figure(1)
    plt.clf()
    plt.plot(exactTime,exactTemp,'r+')
    plt.plot(approxTime,approxTemp)
    theAx=plt.gca()
    theAx.set_xlim([0,10])
    theAx.set_ylim([15,30])
    theAx.set_title('stability')
    plt.show()
   
