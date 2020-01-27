# %%
import numpy as np
import context
from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
a = 5
ax.plot([0, 1], [0, 1])
ax.set_title("the title IIIII")

# %%
from pathlib import Path

print(Path())
