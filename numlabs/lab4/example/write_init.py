"""
   write the initial condition file for the simple oscillator
   example
"""

import json

initialVals={'yinitial': [0.,1.],'t_beg':0.,'t_end':40.,'dt':0.1,'c1':0.,'c2':1.}
initialVals['comment'] = 'written Sep. 29, 2015'
initialVals['plot_title'] = 'simple damped oscillator run 1'

with open('run_1.json','w') as f:
      f.write(json.dumps(initialVals,indent=4))
