"""
 this module uses the algorithms introduced in lab2_functions.py to
 solve weather ballon problem of Lab 1.  We need to integrate two
 variables (height y and velocity u) which are kept in the
 vector 
"""

import numpy as np
from matplotlib import pyplot as plt

def yprime(the_time,yvec,gamma=1,m=1):
    yprime=np.empty_like(yvec)
    yprime[0]=yvec[1]
    yprime[1]= -gamma/m*yvec[0]
    return yprime

def euler(the_times,yvec_init):
    output=np.empty((2,len(the_times)))
    output[:,0]=yvec_init[:]
    for index,the_time in enumerate(the_times[1:]):
        delt=the_times[index+1] - the_times[index]
        output[:,index + 1]=output[:,index] + delt*yprime(the_time,output[:,index])
    return output

def midpoint(the_times,yvec_init):
    output=np.empty((2,len(the_times)))
    output[:,0]=yvec_init[:]
    for index,the_time in enumerate(the_times[1:]):
        delt=the_times[index+1] - the_times[index]
        midguess=delt/2.*yprime(the_time,output[:,index])
        mid_time=the_times[index] + delt/2.
        output[:,index + 1]=output[:,index] + delt*yprime(mid_time,output[:,index]+midguess)
    return output


def leapfrog(the_times,yvec_init):
    output=np.empty((2,len(the_times)))
    output[:,0]=yvec_init[:]
    delt=the_times[1] - the_times[0]
    mid_vals=output[:,0] + delt/2.*yprime(the_times[0],output[:,0])
    for index,the_time in enumerate(the_times[1:]):
        delt=the_times[index+1] - the_times[index]
        mid_time=the_time + delt/2.
        output[:,index + 1]=output[:,index] + delt*yprime(mid_time,mid_vals)
        mid_vals=mid_vals + delt*yprime(the_times[index + 1],output[:,index+1])                             
    return output



if __name__=="__main__":
    the_times=np.linspace(0,20.,100)
    yvec_init=[1,0]
    output_euler=euler(the_times,yvec_init)
    output_mid=midpoint(the_times,yvec_init)
    output_leap=leapfrog(the_times,yvec_init)
    answer=np.sin(the_times)
    plt.close('all')
    plt.style.use('ggplot')
    fig,ax=plt.subplots(1,1)
    ax.plot(the_times,(output_euler[0,:]-answer),label='euler')
    ax.plot(the_times,(output_mid[0,:]-answer),label='midpoint')
    ax.plot(the_times,(output_leap[0,:]-answer),label='leapfrog')
    ax.set(ylim=[-2,2],xlim=[0,20])
    ax.legend(loc='best')
    plt.show()
        
