def jacobian_2(answer):
    '''Gives the responses to the Jacobian quiz: what is the discretization using second order centered differences
    '''

    # dictionary of responses for the jacobian_2 quiz
    response = {'A': 'Try again',
                'B': 'Correct',
                'C': 'The grid we are using is not staggered (like in Lab #7), and so does not have 1/2 points',
                'D': 'Try again',
                'E': 'The a terms have two subscripts on them which means that they should have been differenced twice.  However, there is only a single derivative applied to a',
                'F': 'This is actually the discretization of the first form of the Jacobian we saw (Jacobian: Expansion 1)',
                'G': "Take a look at the hint (if you haven't already) and then try again.",
                'Hint': 'It may help to look at the factors of a inside the derivative ... these are only differenced once, while the b terms are differenced twice.'}
    try:
        return response[answer]
    except KeyError:
        return "Acceptable answers are 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'Hint'"

def jacobian_3(answer):
    '''Gives the responses to the Jacobian quiz: what is the discretization using second order centered differences
    '''

    # dictionary of responses for the jacobian_3 quiz
    response = {'A': 'Try again',
                'B': "Haven't you done the first quiz yet?",
                'C': 'The grid we are using is not staggered (like in Lab #7), and so does not have 1/2 points',
                'D': 'Try again',
                'E': 'Try again',
                'F': 'This is actually the discretization of the first form of the Jacobian we saw (Jacobian: Expansion 1)',
                'G': "You're right!",
                'Hint': 'It may help to look at the factors of b inside the derivative ... these are only differenced once, while the a terms are differenced twice.'}
    try:
        return response[answer]
    except KeyError:
        return "Acceptable answers are 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'Hint'"

