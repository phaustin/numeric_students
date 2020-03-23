# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info,-toc,-latex_envs
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Installs for this notebook:
#
#     pip install contexttimer
#     conda install -c conda-forge numba
#  

# %%
from IPython.display import Image
import contexttimer
import time
import math
from numba import jit
import logging


# %% [markdown]
# # Using numba to speed up python
#
# Compile sections of python code to machine code using the numba "just in time" compiler numba.jit

# %% [markdown]
# ### Timing python code
#
#
# One easy way to tell whether you are utilizing multiple cores is to track the wall clock time measured by [time.perf_counter](https://docs.python.org/3/library/time.html#time.perf_counter) against the total cpu time used by all threads meausred with [time.process_time](https://docs.python.org/3/library/time.html#time.process_time)
#
# I'll organize these two timers using the [contexttimer](https://github.com/brouberol/contexttimer) module.

# %% [markdown]
# #### Define a function that does a lot of computation

# %%
def wait_loop(n):
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
# #### now time it with pure python

# %%
nloops=200
with contexttimer.Timer(time.perf_counter) as pure_wall:
    with contexttimer.Timer(time.process_time) as pure_cpu:
        result=wait_loop(nloops)
print(f'pure python wall time {pure_wall.elapsed} and cpu time {pure_cpu.elapsed}')


# %% [markdown]
# ### Now try this with numba
#
# Numba is a just in time compiler that can turn a subset of python into machine code using the llvm compiler.
#
# Reference:  [Numba documentation](http://numba.pydata.org/numba-doc/dev/index.html)

# %% [markdown]
# ### Make two identical functions: one that releases and one that holds the GIL

# %%
@jit('float64(int64)', nopython=True, nogil=True)
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
@jit('float64(int64)', nopython=True, nogil=False)
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
# ### now time wait\_loop\_withgil

# %%
nloops=500
with contexttimer.Timer(time.perf_counter) as numba_wall:
    with contexttimer.Timer(time.process_time) as numba_cpu:
        result=wait_loop_withgil(nloops)
print(f'numba wall time {numba_wall.elapsed} and cpu time {numba_cpu.elapsed}')
print(f"numba speed-up factor {(pure_wall.elapsed - numba_wall.elapsed)/numba_wall.elapsed}")

# %% [markdown]
# ### not bad, but we're only using one core
