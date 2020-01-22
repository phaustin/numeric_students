def conduction_quiz(answer):
    '''Gives the responses to the conduction quiz: is lambda positive?'''

    # dictionary of responses for conduction quiz
    response = {'True': 'Right!', 'False': 'Sorry.  The idea here is as follows:  Assume that the object is warmer than its surroundings, i.e. T-Ta > 0. We know that the temperature of the object should decrease with time, which translates into dT/dt < 0 in terms of the derivative.  Now, in order that both sides of the differential equation have the same sign, this requires that $\lambda$ > 0.', 'Hint 1': 'Think physically!', 'Hint 2': 'Suppose that the object is warmer than its surroundings (i.e. T-Ta > 0).  Think about what this means for the signs of the terms in the differential equation ....'}
    try:
        return response[answer]
    except KeyError:
        return "Acceptable answers are 'True', 'False', 'Hint 1' or 'Hint 2'"

def interpolation_quiz(answer):
    '''Gives the responses to the interpolation quiz: increasing number of points increases accuracy'''

    # dictionary of responses for the interpolation quiz
    response = {'True': 'This is sometimes true, but not always. See the hint if you want some help.',
                'False': "You're right! When the function to be approximated is smooth (like the function f(x)), then  it is often true that increasing the number of points improves the approximation.  However,  for functions such as g(x) that are not so well-behaved, this may not be the case. This issue of accuracy of approximations will show up again and again in this lab and the rest of the  course.",
                'Hint': ' Go back to the demo and compare the plots when the number of points is 16 and 20 for  function g.'}
    try:
        return response[answer]
    except KeyError:
        return "Acceptable answers are 'True', 'False', or 'Hint'"

def discretization_quiz(answer):
    '''Gives the responses to the discretization quiz: what phrase best describes discretization'''

    # dictionary of responses for the discretization quiz
    response = {'A': "You're on the right track, but this would be closer to a definition of the entire discipline of numerical analysis. Try something a bit more specific.",
                'B': "That's right!",
                'C': "Good guess .... but you're wrong!   We didn't say anything about the discrete Fourier transform in this lab.",
                'D': "It's true that this is an example of discretization, but this is a bit too specific.  There is another definition that fits the term better...",
                'Hint': "Try glancing over the Discretization section of the lab again."}
    try:
        return response[answer]
    except KeyError:
        return "Acceptable answers are 'A', 'B', 'C', 'D' or 'Hint'"  
