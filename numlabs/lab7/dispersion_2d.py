import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def dispersion_2d(disp1, disp2, Rod):
    '''plot 2d dispersion plots for two given dispersion surfaces'''

    k = np.arange(-np.pi,np.pi,0.1)
    l = np.arange(-np.pi,np.pi,0.1)
    kd, ld = np.meshgrid(k, l)

    omegaf1 = disp1(kd, ld, Rod=Rod)
    omegaf2 = disp2(kd, ld, Rod=Rod)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(kd, ld, omegaf1, label='First')
    ax.plot_wireframe(kd, ld, omegaf2, color='g', label='Second')
    ax.set_xlabel('$kd$')
    ax.set_ylabel('$\ell d$')
    ax.set_zlabel('$\omega/f$')
    ax.legend()
    
