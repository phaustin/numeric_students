try:
    from lab5_funs import Integrator
except ImportError:
    import os,sys
    importPath='../../../lab5/lab5_files/python/'
    libdir=os.path.abspath(importPath)
    sys.path.append(libdir)
    from lab5_funs import Integrator


class LorenzInt(Integrator):
    
    def __init__(self,coeffFileName):
        Integrator.__init__(self,coeffFileName)
        i=self.initVars
        i.yinit=np.array([i.x,i.y,i.z])
        i.nVars=len(i.yinit)

    def derivs5(self,y,theTime):
        """y[0]=x, y[1]=y, y[2]=z
        """
        u=self.userVars
        f=np.empty_like(self.initVars.yinit)
        f[0]= u.sigma*(y[1] - y[0])
        f[1] = u.r*y[0] - y[1] - y[0]*y[2]
        f[2] = y[0]*y[1] - u.b*y[2]
        return f


if __name__=="__main__":
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt

    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    theSolver=LorenzInt('lorenz.ini')
    timeVals,yVals,errorList=theSolver.timeloop5Err()
    yVals=np.array(yVals)

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(yVals[:,0],yVals[:,1],yVals[:,2])
    plt.show()
    
