"""

To run:

conda install -c conda-forge pytest

then:

pytest testing_functions.py

This script tests the function bogus_fun by giving it
an input argument of [1,2,3] and checking to see that
it returns [2,4,6] to 3 decimal places

Try changing the expected output and see what happens
"""

from numpy.testing import assert_almost_equal
import numpy as np
import pytest

def bogus_fun(input_list):
    output_list = np.array(input_list)*2.
    return output_list

def test_fun():
   """
   execute unit tests for bogus_fun
   """
   output_list = bogus_fun([1,2,3])
   assert_almost_equal(output_list,[2,5,6], decimal=3)

if __name__ == "__main__":
    np.set_printoptions(precision=4)
    print('testing __file__: {}'.format(__file__))
    pytest.main([__file__, '-vv'])
