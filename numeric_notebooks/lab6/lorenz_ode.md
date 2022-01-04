---
jupytext:
  cell_metadata_filter: -all
  notebook_metadata_filter: all,-language_info,-toc,-latex_envs
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# scipy.integrate.odeint example

To make the animation, do:

`conda install -c conda-forge ffmpeg`

```{code-cell} ipython3
import numpy as np
from scipy import integrate
```

```{code-cell} ipython3
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
```

```{code-cell} ipython3
N_trajectories = 20
```

```{code-cell} ipython3
def lorentz_deriv(coords, t0, sigma=10., beta=8./3, rho=28.0):
    x,y,z = coords
    out = [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    return out
```

```{code-cell} ipython3
# Choose random starting points, uniformly distributed from -15 to 15
np.random.seed(1)
x0 = -15 + 30 * np.random.random((N_trajectories, 3))
```

```{code-cell} ipython3
# Solve for the trajectories
t = np.linspace(0, 4, 1000)
x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t)
                  for x0i in x0])
```

```{code-cell} ipython3
# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off');
```

```{code-cell} ipython3
# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))
```

```{code-cell} ipython3
# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])
```

```{code-cell} ipython3
# prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))
```

```{code-cell} ipython3
# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)
```

```{code-cell} ipython3
# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts
```

```{code-cell} ipython3
# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts
```

```{code-cell} ipython3
# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=500, interval=30, blit=True)
```

```{code-cell} ipython3
# Save as mp4. This requires mplayer or ffmpeg to be installed
filename = 'lorentz_attractor.mp4'
anim.save(filename, fps=15, extra_args=['-vcodec', 'libx264'])
```

```{code-cell} ipython3
display(fig)
```

```{code-cell} ipython3
from IPython.display import Video
Video(filename)
```
