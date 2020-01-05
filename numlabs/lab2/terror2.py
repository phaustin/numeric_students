import matplotlib.pyplot as plt
from numlabs.lab2.lab2_functions import euler,leapfrog,runge,midpoint
import numpy as np

theFuncs={'euler':euler,'leapfrog':leapfrog,'runge':runge,'midpoint':midpoint}

if __name__=="__main__":
    Ta= 20
    To= 30
    tend = 10.0
    theLambda= 0.8
    npts=30
    funChoice='leapfrog'
    funChoice='midpoint'
    funChoice='euler'
    #
    #find the method in the theFuncs dictionary and call it
    #
    approxTime,approxTemp=theFuncs[funChoice](npts,tend,To,Ta,theLambda)
    plt.close('all')
    fig1,ax1=plt.subplots(1,1)
    ax1.plot(approxTime,approxTemp,label=funChoice)
    exactTime=np.empty([npts,],np.float)
    exactTemp=np.empty_like(exactTime)
    for i in np.arange(0,npts):
        exactTime[i] = tend*i/npts
        exactTemp[i] = Ta + (To-Ta)*np.exp(theLambda*exactTime[i])
    ax1.plot(exactTime,exactTemp,'r+',label='exact')
    outdict=dict(deltat=tend/npts,func=funChoice)
    title="exact and approx using {func} with deltat={deltat:5.2g}".format_map(outdict)
    ax1.set(title=title)
    ax1.legend(loc='best')
    fig2,ax2=plt.subplots(1,1)
    difference = exactTemp - approxTemp
    ax2.plot(exactTime,difference)
    title="exact - approx using {func} with deltat={deltat:5.2g}".format_map(outdict)
    ax2.set(title=title)
    plt.show()
