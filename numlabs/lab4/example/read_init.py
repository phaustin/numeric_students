import json
from collections import namedtuple

with open('run_1.json','r') as f:
      init_dict=json.load(f)

print('as a dictionary:\n{}\n'.format(init_dict))

#either use this as a dict or convert to a namedtuple
initvals=namedtuple('initvals','dt c1 c2 t_beg t_end yinitial comment plot_title')
theCoeff=initvals(**init_dict)

print('as a namedtuple:\n{}'.format(theCoeff))
