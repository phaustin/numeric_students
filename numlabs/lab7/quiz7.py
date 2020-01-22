def dispersion_quiz(answer):
    '''Gives the responses to the quiz: What is the dispersion relation for 1-dimensional Poincare Wave?'''

    # dictionary of responses for
    response = {'A': 'Wrong. These are 1-dimensional waves - there is no l (ell).',
                'B': 'Wrong. What about f?',
                'C': 'Correct',
                'D': 'Wrong. Watch your signs'}
    try:
        return response[answer]
    except KeyError:
        return "Acceptable answers are 'A', 'B', 'C' or 'D'"
