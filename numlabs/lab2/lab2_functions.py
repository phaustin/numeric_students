"""
    This module contains four finite differencing functions
    with the same interface:  euler, beuler, leapfrog and runge

    Parameters
    ----------

     npts : integer
        number of timesteps
     tend : float
        stopping time (seconds)
     To : float
        inital temperature of object (deg C)
     Ta : float
        ambient temperature of air (deg C)
     theLambda : float
        thermal diffusion coefficient (:math:`(s^{-1})`)

    Returns
    -------

     (theTime,theTemp) : (array,array)
         two arrays each of length npts, with time and temperature

"""

import numpy as np

def heat(theTemp,Ta,theLambda):
    out=theLambda*(theTemp-Ta)
    return out

def euler(npts,tend,To,Ta,theLambda):
    """
     Integrate the thermal diffusion equation
     using forward euler
    
     Parameters
     ----------
     npts : integer
        number of timesteps
     tend : float
        stopping time (seconds)
     To : float
        inital temperature of object (deg C)
     Ta : float
        ambient temperature of air (deg C)
     theLambda : float
        thermal diffusion coefficient (:math:`(s^{-1})`)

     Returns
     -------
     (theTime,theTemp) : (array,array)
         two arrays each of length npts, with time and temperature
    """

    theTemp=np.empty([npts,],np.float64)
    theTemp[0]=To
    dt=tend/npts
    theTime=np.empty_like(theTemp)
    theTime[0]=0.
    for timeStep in np.arange(1,npts):
        theTime[timeStep]=theTime[timeStep-1] + dt
        theTemp[timeStep]=theTemp[timeStep-1] + \
            heat(theTemp[timeStep-1],Ta, theLambda)*dt
    return (theTime,theTemp)

def beuler(npts,tend,To,Ta,theLambda):
    #pdb.set_trace()
    dt=tend/npts;
    theTemp=np.empty([npts,],np.float64)
    theTemp[0]=To
    theTime=np.zeros_like(theTemp)
    for timeStep in np.arange(1,npts):
        theTime[timeStep]=theTime[timeStep-1] + dt
        theTemp[timeStep] = (theTemp[timeStep-1]-theLambda*dt*Ta)/(1-theLambda*dt)
    return (theTime,theTemp)

def leapfrog (npts,tend,To,Ta,theLambda):
    dt=tend/npts
    theTemp=np.empty([npts,],np.float64)
    theTemp[0]=To
    theTime=np.empty_like(theTemp)
# estimate first step by forward euler as need two steps to do leapfrog
    theTemp[1] = To + heat(To,Ta,theLambda)*dt
    theTime[1] = dt
# correct first step by estimating the temperature at a half-step
    Th = To + 0.5*(theTemp[1]-To)
    theTemp[1] = To + heat(Th,Ta,theLambda)*dt
    for timeStep in np.arange(2,npts):
        theTime[timeStep]=theTime[timeStep-1] + dt
        theTemp[timeStep] = theTemp[timeStep-2]+\
                            heat(theTemp[timeStep-1],Ta,theLambda)*2.*dt
    return (theTime,theTemp)

def midpoint (npts,tend,To,Ta,theLambda):
    dt=tend/npts
    theTemp=np.empty([npts,],np.float64)
    theTemp[0]=To
    theTime=np.empty_like(theTemp)
# estimate first step by forward euler as need two steps to do leapfrog
    theTemp[1] = To + heat(To,Ta,theLambda)*dt
    theTime[1] = dt
# correct first step by estimating the temperature at a half-step
    Th = To + 0.5*(theTemp[1]-To)
    theTemp[1] = To + heat(Th,Ta,theLambda)*dt
    for timeStep in np.arange(2,npts):
        theTime[timeStep]=theTime[timeStep-1] + dt
        theTemp[timeStep] = theTemp[timeStep-2]+\
                            heat(theTemp[timeStep-1],Ta,theLambda)*2.*dt
    return (theTime,theTemp)


def runge(npts,tend,To,Ta,theLambda):
    dt=tend/npts
    theTemp=np.empty([npts,],np.float64)
    theTemp[0]=To
    theTime=np.empty_like(theTemp)
    theTime[0] = 0
    for i in np.arange(1,npts):
        k1 = dt * heat(theTemp[i-1],Ta,theLambda)
        k2 = dt * heat(theTemp[i-1] + (0.5 * k1),Ta,theLambda)
        k3 = dt * heat(theTemp[i-1] + (0.5 * k2),Ta,theLambda)
        k4 = dt * heat(theTemp[i-1] +  k3,Ta,theLambda)
        theTemp[i] = theTemp[i-1] + (1.0/6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)
        theTime[i] = theTime[i-1]+dt
    return (theTime,theTemp)






