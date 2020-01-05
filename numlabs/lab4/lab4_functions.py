from __future__ import division
import numpy as np
from collections import namedtuple 

def initinter41(valueDict):
# function to initialize variable for example 1
    initvals=namedtuple('initvals','dt c1 c2 c3 t_beg t_end yinitial')
    theCoeff=initvals(**valueDict)
    return theCoeff

def derivsinter41(coeff, y, theTime):
    f = coeff.c1*y + coeff.c2*theTime+coeff.c3
    return f

def eulerinter41(coeff,y,theTime):
    y=y + coeff.dt*derivsinter41(coeff,y,theTime)
    return y

def midpointinter41(coeff, y,theTime):
    midy=y + 0.5 * coeff.dt * derivsinter41(coeff,y,theTime)
    y = y + coeff.dt*derivsinter41(coeff,midy,theTime+0.5*coeff.dt)
    return y

def rk4ODEinter41(coeff, y, theTime):
  k1 = coeff.dt * derivsinter41(coeff,y,theTime)
  k2 = coeff.dt * derivsinter41(coeff,y + (0.5 * k1),theTime+0.5*coeff.dt)
  k3 = coeff.dt * derivsinter41(coeff,y + (0.5 * k2),theTime+0.5*coeff.dt)
  k4 = coeff.dt * derivsinter41(coeff,y +  k3,theTime+coeff.dt)
  y = y + (1.0/6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)
  return y

def rkckODEinter41(coeff,yold,told):

## initialize the Cash-Karp coefficients
## defined in the tableau in lab 4,
## section 3.5
  a = np.array([.2, 0.3, 0.6, 1.0, 0.875])
  c1 = np.array([37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0])
  c2= np.array([2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0,
       277.0/14336.0, .25])
  b=np.empty([5,5],'float')
  c2 = c1 - c2
  b[0,0] =0.2 
  b[1,0]= 3.0/40.0 
  b[1,1]=9.0/40.0
  b[2,0]=0.3 
  b[2,1]=-0.9 
  b[2,2]=1.2
  b[3,0]=-11.0/54.0 
  b[3,1]=2.5 
  b[3,2]=-70.0/27.0 
  b[3,3]=35.0/27.0
  b[4,0]=1631.0/55296.0 
  b[4,1]=175.0/512.0 
  b[4,2]=575.0/13824.0
  b[4,3]=44275.0/110592.0 
  b[4,4]=253.0/4096.0

# set up arrays
  
  derivArray=np.empty([6],'float')
  ynext=np.zeros_like(yold) 
  bsum=np.zeros_like(yold) 
  derivArray[0]=derivsinter41(coeff,yold,told)
  
# calculate step
  
  y=yold
  for i in np.arange(5):
    bsum=0.
    for j in np.arange(i+1): 
        bsum=bsum + b[i,j]*derivArray[j]
    derivArray[i+1]=derivsinter41(coeff,y + coeff.dt*bsum,told + a[i]*coeff.dt)
    ynext = ynext + c1[i]*derivArray[i]
  y = y + coeff.dt*(ynext + c1[5]*derivArray[5])
  return y


