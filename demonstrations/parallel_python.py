# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: all,-language_info,-toc,-latex_envs
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parallel computation
#
# A quick demonstration of how to run parallel jobs on multiple cores and or multiple threads.
#
# Install:
#
# ```
# conda install -c conda-forge joblib
# ```
#

# %%
from IPython.display import Image
import contexttimer
import time
import math
from numba import jit
import threading
from joblib import Parallel
import logging
import pprint
pp = pprint.PrettyPrinter(indent=4)


# %% [markdown]
# ## Threads and processes in Python
#
# From [Wikipedia](https://en.wikipedia.org/wiki/Thread_(computing)):
#
# >"In computer science, a thread of execution is the smallest sequence of programmed instructions that can be managed independently by a scheduler, which is typically a part of the operating system.[1] The implementation of threads and processes differs between operating systems, but in most cases a thread is a component of a process. Multiple threads can exist within one process, executing concurrently and sharing resources such as memory, while different processes do not share these resources. In particular, the threads of a process share its executable code and the values of its variables at any given time."
#
#
#
#
# [Reference: Thomas Moreau and Olivier Griesel, PyParis 2017 [Mor2017]](https://tommoral.github.io/talks/pyparis17/#1)
#
# ### Python global intepreter lock
#
# 1. Motivation: python objects (lists, dicts, sets, etc.) manage their own memory by storing a counter that keeps track of how many copies of an object are in use.  Memory is reclaimed when that counter goes to zero.
#
# 1. Having a globally available reference count makes it simple for Python extensions to create, modify and share python objects.
#
# 1. To avoid memory corruption, a python process will only allow 1 thread at any given moment to run python code.  Any thread that wants to access python objects in that process needs to acquire the global interpreter lock (GIL).
#
# 1. A python extension written in C, C++ or numba is free to release the GIL, provided it doesn't create, destroy or modify any python objects.  For example: numpy, pandas, scipy.ndimage, scipy.integrate.quadrature all release the GIL
#
# 1. Many python standard library input/output routines (file reading, networking) also release the GIL
#
# 1. On the other hand:  hdf5, and therefore h5py and netCDF4, don't release the GIL and are single threaded.
#
# 1. Python comes with many libraries to manage both processes and threads.
#
#
# ### Thread scheduling
#
# If multiple threads are present in a python process, the python intepreter releases the GIL at specified intervals (5 miliseconds default) to allow them to execute:

# %% [markdown]
#
#
# <img src='images/morreau1.png'>
#
#
# #### Note that these three threads are taking turns, resulting in a computation that runs slightly slower (because of overhead) than running on a single thread
#
# ### Releasing the GIL
#
# If the computation running on the thread has released the GIL, then it can run independently of other threads in the process.  Execution of these threads are scheduled by the operating system along with all the other threads and processes on the system.
#  
# In particular, basic computation functions in Numpy, like (\__add\__ (+), \__subtract\__ (-) etc. release the GIL, as well as universal math functions like cos, sin etc.

# %% [markdown]
#
# <img srce='images/morreau2.png'>
#
# <img src='images/morreau3.png'>

# %% [markdown]
#  ## Creating a thread pool with joblib
#
#
# [joblib](https://pythonhosted.org/joblib/index.html) Provides the best way to run naively parallel jobs on multiple threads or processes in python.
#
# * It integrates seamlessly with [dask](http://distributed.readthedocs.io/en/latest/joblib.html)
#  and [scikit-learn](http://scikit-learn.org/stable/modules/model_persistence.html)
#   
# * It has a much better multiprocessing library than standard python: [loky](https://github.com/tomMoral/loky)
#
# * To use it, create a Parallel object that runs a list of functions, where each function is part of a tuple that specifies the arguments and keywords (if any)
#
#
# ### Our functions from last week's numba notebook 

# %%
@jit('float64(int64)', nopython=True, nogil=True)  #release the GIL!
def wait_loop_nogil(n):
    """
    Function under test.
    """
    for m in range(n):
        for l in range(m):
            for j in range(l):
                for i in range(j):
                    i=i+4
                    out=math.sqrt(i)
                    out=out**2.
    return out


# %%
@jit('float64(int64)', nopython=True, nogil=False) #hold the GIL
def wait_loop_withgil(n):
    """
    Function under test.
    """
    for m in range(n):
        for l in range(m):
            for j in range(l):
                for i in range(j):
                    i=i+4
                    out=math.sqrt(i)
                    out=out**2.
    return out



# %% [markdown]
# ### Setup logging so we can know what process and thread we are running

# %% [markdown]
# ### Create  find_ids to print thread and process ids, and one to run the wait_for loop
#  
# * Important point -- the logging module is **threadsafe**
#
#

# %%
def find_ids():
    logging.debug('debug logging: ')
    
logging.basicConfig(level=logging.INFO,
                    format='%(message)s %(threadName)s %(processName)s',
                    )


# %% [markdown]
# ### Submit 6 jobs queued on 3 processors
#
# ### First get a set of functions that report back their thread id
#
# The cell below creates a 3 item list which will
# run find_ids three separate times and report back the 
# thread that it is running on.

# %%
njobs=6
nprocs=3
thread_id_jobs =[(find_ids,[],{}) for i in range(nprocs)]


# %% [markdown]
# ### Next get a set of functions that do actual work
#
# This cell creates a 6 item list that contains 6 numba functions
# with each function taking an argument of 1250

# %%
nloops=1250
calc_jobs=[(wait_loop_nogil,[nloops],{}) for i in range(njobs)]
pp.pprint(calc_jobs)


# %% [markdown]
# ## Run 1:  Nogil jobs, multithreaded
#
# Note that we have 6 jobs running on 3 threads

# %%
with contexttimer.Timer(time.perf_counter) as wall:
    with contexttimer.Timer(time.process_time) as cpu:
        with Parallel(n_jobs=nprocs,backend='threading') as parallel:
            parallel(thread_id_jobs)
            results=parallel(calc_jobs)
print(results)
print(f'wall time {wall.elapsed} and cpu time {cpu.elapsed}')


# * Each job was run on a different thread but in the same process
# 
# * Note that the cpu time is larger than the wall time, confirming that we've release the GIL.
# 


# %% [markdown]
#  ### Now repeat this holding the GIL
#  
#  No difference, because we are still on the main process

# %%
calc_jobs=[(wait_loop_withgil,[nloops],{}) for i in range(njobs)]
with contexttimer.Timer(time.perf_counter) as wall:
    with contexttimer.Timer(time.process_time) as cpu:
        with Parallel(n_jobs=nprocs,backend='threading') as parallel:
            parallel(thread_id_jobs)
            results=parallel(calc_jobs)
print(results)
print(f'wall time {wall.elapsed} and cpu time {cpu.elapsed}')


# ** Note that the speed is the same as if we ran on a single CPU **

# ### Now repeat with processes instead of threads

# %%
calc_jobs=[(wait_loop_withgil,[nloops],{}) for i in range(njobs)]
with contexttimer.Timer(time.perf_counter) as wall:
    with contexttimer.Timer(time.process_time) as cpu:
        with Parallel(n_jobs=nprocs,backend='loky') as parallel:
            parallel(thread_id_jobs)
            results=parallel(calc_jobs)
        print(results)
print(f'wall time {wall.elapsed} and cpu time {cpu.elapsed}')

# %% [markdown]
# ### how do you explain the tiny cpu time? 
#
# ###  Summary
#  
# 1.  For simple functions without Python code, Numba can release the GIL and you can get the benefit of multiple threads
#  
# 1. The joblib library can be used to queue dozens of jobs onto a specified number of processes or threads
#  
# 1. A process pool can execute pure python routines, but all data has to be copied to and from each process.

# %%




