""" Ported by Vlad Popa from Matlab code for EOSC 511, Laboratory 8 files,
    March 2011.  Ported to Python 3 by SEA in 2015, removed weave and indexed
    instead of looping

    To run:
    1) inside a notebook or spyder, returns psi for you to plot as you want
    import numlabs.lab8.qg as qg
    psi = qg(time, loop=False)
    2) from the command line, returns a plot for you
    python qg.py time False
    
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

def param():
    # set the physical parameters of the system
    Av = 5.e-2           # vertical eddy viscosity (m^2/s)
    H = 500.               # depth (m)
    rho = 1.0e3            # density of water (kg/m^3)
    latitude = 45.         # for calculating parameters of the beta-plane (deg)
    tau_max = 0.2          # wind stress maximum (kg m/s^2)
    b = 2.0e6              # width of the ocean (m)
    a = 2.0e6              # N-S extent of the ocean (m)

    # necessary constants
    omega =  7.272205e-05  # rotation rate of the Earth (/s)
    earth_radius = 6.4e+6  # radius of the Earth (m)

    # calculate derived parameters
    theta = latitude/360*2*np.pi               # convert degrees to radians
    beta = 2*omega*np.cos(theta)/earth_radius  # beta=df/dy
    f0 = 2*omega*np.sin(theta)                 # Coriolis parameter
    kappa = np.sqrt(Av*f0/2)/H                 # Ekman number = delta/H
    boundary_layer_width_approx = kappa/beta   # estimate of BL width (m)

    # calculation the three nondimensional coefficients
    U0 = tau_max/(b*beta*rho*H)
    epsilon = U0/(b*b*beta)
    wind = -1.
    vis = kappa/(beta*b)
    time = 1./(beta*b)

    # display physical parameters
    print ("Physical Parameters:")
    print ("a = ", a)
    print ("b = ", b)
    print ("epsilon = ", epsilon)
    print ("wind = ", wind)
    print ("vis = ", vis)
    print ("time = ", time)
    print ("boundary_layer_width_approx = ", boundary_layer_width_approx)

    return (b, a, epsilon, wind, vis, time)

def numer_init():
    # set up domain
    nx = 30            # number of points in x-direction
    dx = 1./(nx-1)     # size of non-dimensional grid points
    ny = int(1./dx+1)  # number of points in y-direction

    # set time step
    dt = 43.2e3  # time step  

    # set up the parameters for the relaxation scheme
    tol = 0.5e-2   # error tolerance
    maxiter = 50     # maximum number of interation
    coeff = 1.  # relaxation coefficient

    # display simulation parameters
    print ("\nSimulation Parameters:")
    print ("nx = ", nx)
    print ("dx = ", dx)
    print ("ny = ", ny)
    print ("dt = ", dt)
    print ("maximum iterations = ", maxiter)
    print ("tolerance =", tol )
    print ("coeff = ", coeff)

    return (nx, dx, ny, dt, tol, maxiter, coeff)

def vis(psi, nx, ny):
    visc = np.zeros_like(psi)
    
    # loop version
    ## for i in range(1, nx-1):
    ##     for j in range(1, ny-1):
    ##         visc[i,j] = psi[i+1,j]+psi[i-1,j]+psi[i,j+1]+psi[i,j-1]-4*psi[i,j]

    # indexing version
    si = 1; ei = nx-1 # start and end i indices
    sj = 1; ej = ny-1 # start and end j indices
    visc[si:ei, sj:ej] = (
        psi[si+1:ei+1, sj:ej] + psi[si-1:ei-1, sj:ej] +
        psi[si:ei, sj+1:ej+1] + psi[si:ei, sj-1:ej-1]
        - 4 * psi[si:ei, sj:ej])

    return visc

def mybeta(psi,nx,ny):
    beta = np.zeros_like(psi)

    # loop version
    ## for i in range(1,nx-1):
    ##     for j in range(1,ny-1):
    ##         beta[i,j] = psi[i+1,j]-psi[i-1,j]

    # indexing version
    si = 1; ei = nx-1 # start and end i indices
    sj = 1; ej = ny-1 # start and end j indices
    beta[si:ei, sj:ej] = psi[si+1:ei+1, sj:ej] - psi[si-1:ei-1, sj:ej]

    return beta

def wind(nx, ny):
    windy = np.zeros((nx, ny))
                     
    # loop version
    ## for i in range(0,nx):
    ##    for j in range(0,ny):
    ##        # fit one negative cosine curve from southern boundary to northern
    ##        tau[i,j] = -np.cos(np.pi*(j-0.5)/(ny-2))

    ## for i in range(1,nx-1):
    ##    for j in range(1,ny-1):
    ##        windy[i,j] = -tau[i,j+1]+tau[i,j-1]
    
    # indexing version
    den = 1./(ny-2)
    si = 1; ei = nx-1 # start and end i indices
    sj = 1; ej = ny-1 # start and end j indices
    windy [si:ei, sj:ej] = - (
        np.cos(np.pi*(np.arange(sj, ej)+0.5) * den) -
        np.cos(np.pi*(np.arange(sj, ej)-1.5) * den)
        )
    
    return windy

def jac(psi, vis, nx, ny):
    jaco = np.zeros_like(psi)

    # loop based python version of code
    ## for i in range(1,nx-1):
    ##     for j in range(1,ny-1):
    ##         # Arakawa Jacobian
    ##         jaco[i,j] =((psi[i+1,j]-psi[i-1,j])*(vis[i,j+1]-vis[i,j-1])- \
    ##                     (psi[i,j+1]-psi[i,j-1])*(vis[i+1,j]-vis[i-1,j])+ \
    ##                     psi[i+1,j]*(vis[i+1,j+1]-vis[i+1,j-1])-psi[i-1,j]* \
    ##                     (vis[i-1,j+1]-vis[i-1,j-1])-psi[i,j+1]* \
    ##                     (vis[i+1,j+1]-vis[i-1,j+1])+psi[i,j-1]* \
    ##                     (vis[i+1,j-1]-vis[i-1,j-1])+vis[i,j+1]* \
    ##                     (psi[i+1,j+1]-psi[i-1,j+1])-vis[i,j-1]* \
    ##                     (psi[i+1,j-1]-psi[i-1,j-1])-vis[i+1,j]* \
    ##                     (psi[i+1,j+1]-psi[i+1,j-1])+vis[i-1,j]* \
    ##                     (psi[i-1,j+1]-psi[i-1,j-1]))*0.33333333
    
    # indexing version
    si = 1; ei = nx-1 # start and end i indices
    sj = 1; ej = ny-1 # start and end j indices
    jaco[si:ei, sj:ej] = (
        (psi[si+1:ei+1, sj:ej] - psi[si-1:ei-1, sj:ej]) *
        (vis[si:ei, sj+1:ej+1] - vis[si:ei, sj-1:ej-1]) -
        (psi[si:ei, sj+1:ej+1] - psi[si:ei, sj-1:ej-1]) *
        (vis[si+1:ei+1, sj:ej] - vis[si-1:ei-1, sj:ej]) +
        
        (psi[si+1:ei+1, sj:ej]) *
        (vis[si+1:ei+1, sj+1:ej+1] - vis[si+1:ei+1, sj-1:ej-1]) -
        (psi[si-1:ei-1, sj:ej]) *
        (vis[si-1:ei-1, sj+1:ej+1] - vis[si-1:ei-1, sj-1:ej-1]) -
        (psi[si:ei, sj+1:ej+1]) *
        (vis[si+1:ei+1, sj+1:ej+1] - vis[si-1:ei-1, sj+1:ej+1]) +
        (psi[si:ei, sj-1:ej-1]) *
        (vis[si+1:ei+1, sj-1:ej-1] - vis[si-1:ei-1, sj-1:ej-1]) +
        
        (vis[si:ei, sj+1:ej+1]) *
        (psi[si+1:ei+1, sj+1:ej+1] - psi[si-1:ei-1, sj+1:ej+1]) -
        (vis[si:ei, sj-1:ej-1]) *
        (psi[si+1:ei+1, sj-1:ej-1] - psi[si-1:ei-1, sj-1:ej-1]) -
        (vis[si+1:ei+1, sj:ej]) *
        (psi[si+1:ei+1, sj+1:ej+1] - psi[si+1:ei+1, sj-1:ej-1]) +
        (vis[si-1:ei-1, sj:ej]) *
        (psi[si-1:ei-1, sj+1:ej+1] - psi[si-1:ei-1, sj-1:ej-1])
        ) / 3.
    
    return jaco


def chi(psi, vis_curr, vis_prev, nx, ny, dx, epsilon, wind_par, vis_par):
    # calculate right hand side

    beta_term = mybeta(psi, nx, ny)
    wind_term = wind(nx, ny)
    jac_term = jac(psi, vis_curr, nx, ny)
    d = 1./dx

    rhs = (- 0.5*d*beta_term 
           - epsilon*0.25*d*d*d*d*jac_term
           + wind_par*0.5*d*wind_term
           - vis_par*d*d*vis_prev
          )
    
    return rhs

def relax(rhs, chi_prev, dx, nx, ny, r_coeff, tol, max_count, loop):
    chi = np.copy(chi_prev)
    r = np.zeros_like(chi_prev)

    rr = 1e50
    count = 0
    
    si = 1; ei = nx-1 # start and end i indices
    sj = 1; ej = ny-1 # start and end j indices

    while (rr > tol) & (count < max_count): 
    
        if loop:  # loop based python version of code
            r_max = 0.
            chi_max = 0.
    
            for i in range(1, nx-1):
                 for j in range(1, ny-1):
                     r[i,j] = (rhs[i,j] * dx * dx * 0.25 - 
                            ((chi[i+1, j] + chi[i, j+1] 
                              + chi[i-1, j] + chi[i, j-1]) * 0.25 - chi[i,j]) )
                     if np.abs(chi[i,j]) > chi_max:
                         chi_max = np.abs(chi[i,j])
                     if np.abs(r[i,j]) > r_max:
                         r_max = np.abs(r[i,j])
                     chi[i,j] = chi[i,j] - r_coeff*r[i,j]
        else:    # index loop to use built in c-processing            
            r[si:ei, sj:ej] = rhs[si:ei, sj:ej] * dx * dx * 0.25 - (
                ( chi[si+1:ei+1, sj:ej]
                + chi[si:ei, sj+1:ej+1]
                + chi[si-1:ei-1, sj:ej]
                + chi[si:ei, sj-1:ej-1] ) * 0.25 
                - chi[si:ei, sj:ej] )
        
            chi_max = max(np.max(chi), -np.min(chi))
            r_max = max(np.max(r), -np.min(r))
            chi = chi - r_coeff * r
        
        if (chi_max==0): 
            rr = 1e50 
        else:
            rr=r_max / chi_max
        count = count + 1
        
    return (chi, count)

def qg(totaltime, loop=False):
    
    # initialize the physical parameters
    (pb, pa, pepsilon, pwind, pvis, ptime) = param()
    # initialize the numerical parameters
    (nnx, ndx, nny, ndt, ntol, nmax, ncoeff) = numer_init()

    # initialize the arrays (need 2 because chi depends on psi at 2 time steps)
    psi_1 = np.zeros((nnx,nny))
    psi_2 = np.zeros((nnx,nny))

    vis_prev = np.zeros((nnx,nny))
    vis_curr = np.zeros((nnx,nny))

    chii = np.zeros((nnx,nny))
    chi_prev = np.zeros((nnx,nny))

    # non-dimensionalize time
    totaltime = totaltime/ptime
    dt = ndt/ptime

    # start time loop
    t = 0
    count = 0
    count_total = 0

    # loop
    while (t < totaltime):
        # write some stuff on the screen so you know code is working
        t = t + dt
        
        # update viscosity
        vis_prev = vis_curr
        vis_curr = vis(psi_1, nnx, nny)
        
        # find chi, take a step
        rhs = chi(psi_1, vis_curr, vis_prev, nnx, nny, ndx, pepsilon, pwind, pvis)
        (chii, c) = relax(rhs, chi_prev, ndx, nnx, nny, ncoeff, ntol, nmax, loop)
        psi_2 = psi_2 + dt*chii
        chi_prev = chii
        count_total = count_total + c
        
        # do exactly the same thing again with opposite psi arrays
        t = t + dt
        
        vis_prev = vis_curr
        vis_curr = vis(psi_2, nnx, nny)
        rhs = chi(psi_2, vis_curr, vis_prev, nnx, nny, ndx, pepsilon, pwind, pvis)
        (chii, c) = relax(rhs, chi_prev, ndx, nnx, nny, ncoeff, ntol, nmax, loop)
        psi_1 = psi_1 + dt*chii
        chi_prev = chii
        count_total = count_total + c
        count = count + 1
        
    return psi_1


def main(args):
    if len(args) > 3:
        print ('Usage: qg n_time loop=False')
        print ('n_time = time in seconds; default is days = 10*86400')
        print ('loop = whether to loop or index, default = False')
    else:
        if len(args) == 1:
            # Default to 10 days, loop false
            psi = qg(86400*10)
        elif len(args) == 2:
            psi = qg(float(args[1]))
        else:
            loop = True if args[2].lower() == 'true' else False
            psi = qg(float(args[1]), loop)
        fig, ax = plt.subplots(1, 1, figsize=(10,8))
        mesh = ax.contourf(np.transpose(psi), cmap='copper')
        fig.colorbar(mesh)
        plt.show()
    

if __name__ == '__main__':
    main(sys.argv)
           

