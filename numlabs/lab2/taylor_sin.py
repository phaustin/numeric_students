import numpy as np
from math import factorial
from matplotlib import pyplot as plt

def sin_taylor(x):
    """
    lab 2 eq: taylor
    return the first 20 terms of the taylor series
    for sin(x) evaluated about x=0

    works for scalar and np.array values for x
    """
    term_list = []
    the_sign=1.
    for k in range(20):
        term_coeff = (2 * k) + 1
        fac = the_sign*factorial(term_coeff)
        term_list.append(x**(term_coeff) / fac)
        the_sign *= -1
    return term_list

#
# if we import taylor_sin as a module, then __name__ will
# be the module name "taylor_sin" and this code will not run
#
# if we run this module from the command line, then __name__
# will be "__main__" and a plot will be generated
#
if __name__ == "__main__":

    plt.style.use('ggplot')
    plt.close('all')
    
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(-10, 10, 100)
    the_terms = sin_taylor(x)
    
    sum_terms = np.zeros_like(x)
    for k,a_term in enumerate(the_terms[:6]):
        sum_terms = sum_terms + a_term
        label='$P_{{{:d}}}$'.format(2*k+1)
        ax.plot(x,sum_terms,label=label)
        
    ax.plot(x,np.sin(x),label='exact')
    ax.set(ylim=[-4,4])
    ax.legend()
    fig.savefig('taylor.png')
    plt.show()
