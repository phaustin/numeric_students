import numpy as np
from matplotlib import pyplot as plt
x=np.linspace(-10,30,50)
y=6*x - x**2/3.
fig=plt.figure(1)
plt.plot(x,y)
plt.show()
