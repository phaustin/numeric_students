"""
to raise a figure that's buried under other windows:

from numlabs.show_figure import show_plot as sp

> sp()

"""
from matplotlib import pyplot as plt
import warnings
import matplotlib.cbook
#http://stackoverflow.com/questions/24502500/python-matplotlib-getting-ride-of-matplotlib-mpl-warning
#suppress a gui deprecation warning
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

#http://stackoverflow.com/questions/8202228/make-matplotlib-plotting-window-pop-up-as-the-active-one
def show_plot(figure_id=None):    
    if figure_id is not None:
        fig = plt.figure(num=figure_id)
    else:
        fig = plt.gcf()

    plt.show()
    plt.pause(1e-9)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()

