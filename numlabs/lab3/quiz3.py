'''
   quiz module for lab 3
'''

def matrix_quiz(answer):
    '''Gives the responses to the matrix quiz: which matrix represents the given set of equations
    '''

    # dictionary of responses for matrix quiz
    response = {'A': "Don't get your columns and rows mixed up! Try again.",
                'B': 'You forgot the augmented column. Try again.',
                'C': 'Bingo! Note that even if the rows are mixed up, the matrix is still correct.',
                'D': 'Watch your signs. Try again.'}
    try:
        return response[answer]
    except KeyError:
        return "Acceptable answers are 'A', 'B', 'C' or 'D'"
