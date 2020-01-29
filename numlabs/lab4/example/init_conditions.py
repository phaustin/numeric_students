"""
   default conditions for this problem
"""
from collections import namedtuple

initialVals = {
    'yinitial': [0., 1.],
    't_beg': 0.,
    't_end': 40.,
    'dt': 0.1,
    'c1': 0.,
    'c2': 1.
}
initialVals['comment'] = 'written Jan 29,2020'
initialVals['plot_title'] = 'simple damped oscillator run 1'


def get_init():
    #
    # convert dictionary to namedtuple
    # and return it  
    #
    initvals = namedtuple('initvals',
                          'dt c1 c2 t_beg t_end yinitial comment plot_title')
    theCoeff = initvals(**initialVals)
    return theCoeff
