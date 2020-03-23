# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all,-language_info,-toc,-latex_envs
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Installation
#
#
# ##  Install the ode module from scipy 1.0 beta
#
#     git clone https://github.com/phaustin/eos_integrate.git
#     cd eos_integrate
#     pip install .
#     
#     
# ## Install contexttimer and joblib
#
#     pip install contexttimer
#     conda install joblib
#     

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from eos_integrate import solve_ivp, DenseOutput

# %% [markdown]
# # Van der Pol oscillator

# %% [markdown]
# The system for Van der Pol oscillator is given as:
# $$
# y_1' = y_2, \\
# y_2' = \mu (1 - y_1^2) y_2 - y_1 \\
# y_1(0) = 2, \quad y_2(0) = 0
# $$
# It becomes stiff for high values of $\mu$, meaning that regions of rapid transition are followed by regions where the solution varies slowly. Explicit methods either diverge or make prohibitevely many steps for stiff problems, thus implicit methods should be used. Our function `solve_ivp` implements a one-step fully implicit Runge-Kutta method of Radau II A family.

# %% [markdown]
# We want to solve this for a range of mu values

# %% [markdown]
# ## Use [functools.partial](https://docs.python.org/3/library/functools.html#functools.partial)
# to turn a derivs function with arguments (mu, t, y) into a function of (t,y)

# %%
import functools
import contexttimer
import time

def fun(mu, t, y):
    return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]
#
# create a new function specialize for mu=150
#
mu=150
derivs = functools.partial(fun,mu)           

# %% [markdown]
# ## Use [contexttimer](https://pypi.python.org/pypi/contexttimer/0.3.3) 
# to keep track of wall clock and execution time

# %%
tstart=0
tstop=50
y0init=2
y1init=0
with contexttimer.Timer(time.perf_counter) as wall:
    with contexttimer.Timer(time.process_time) as cpu:
        res = solve_ivp(derivs, [tstart,tstop], [y0init, y1init], method='RK45')
print((f'wall time is {wall.elapsed:5.2f} seconds '
        f'and cpu time is {cpu.elapsed:5.2f} seconds'))

# %%
fig,ax=plt.subplots(1,1)
ax.plot(res.t, res.y[0])
ax.plot(res.t, res.y[0], '.');
ax.set(xlabel='time (seconds)',ylabel='y[0]');


# %% [markdown]
# # Using multiple cores with joblib

# %% [markdown]
# Suppose I want to solve the Van der Pol equation for many different mu values.
# My macbook has two cores -- that should allow me to run multiple jobs more quickly.
# The easiest way to do this kind of simple parallel processing in python is to use
# the [joblib](https://pythonhosted.org/joblib/parallel.html#common-usage) module.
#
# joblib requires that the jobs to be run be submitted as a list, with each job consisting of
# three items:  
#
# 1) the function to be run
# 2) any required positional arguments to the function
# 3) any optional keyword arguments to the function

# %%
#
#  wrap the solver for joblib
#
#  keep the derivs function the same as above
#
def fun(mu, t, y):
    return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]
#
#  here is the "all in one" function that joblib needs to run to 
#  integrate the ode
#
def joblib_solver(fun,mu,tstart,tstop,y0init,y1init,method='RK45'):
    derivs = functools.partial(fun,mu) 
    res = solve_ivp(derivs, [tstart,tstop], [y0init, y1init], method=method)
    return res


# %%
#
# now create a list of 15 different jobs for joblib, 
# each with a different
# mu between 150 and 300 but with everything else the same
#
muvals=np.linspace(150,300,15)
joblist=[]
for mu in muvals:
    positional_args=[fun,mu,tstart,tstop,y0init,y1init]
    keyword_args={'method':'Radau'}
    joblist.append((joblib_solver,positional_args,keyword_args))

# %% [markdown]
# How long does it take to run 15 jobs on 1 core?  How about 2 cores?
#
# Try changing 'RK45'  to 'Radau' in the cell above and see what the timing looks like

# %%
from joblib import Parallel
ncores=2
with contexttimer.Timer(time.perf_counter) as wall:
    with Parallel(n_jobs=ncores,backend='threading') as parallel:
        results=parallel(joblist)
print(f'wall time  with {ncores} cores is {wall.elapsed:5.2f} seconds')

# %% [markdown]
# Here is what the first result object looks like.

# %%
results[0]

# %% [markdown]
# Plot all 15 results with a legend

# %%
fig,ax=plt.subplots(1,1,figsize=[8,8])
for result,mu in zip(results,muvals):
    ax.plot(result.t,result.y[0],label=f'mu = {mu:5.1f}')
ax.legend();

# %%
