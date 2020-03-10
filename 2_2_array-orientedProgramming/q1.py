"""In this exercise, you will program a helper function, which will normalise
the data to have zero average value and a variance of one for each column.
This is done by first calculating the column averages, and then subtracting the
column average from all values in that column. Then, all values in a column are
divided by the standard deviation, this will complete the standardisation.
Print the array.
To format the output, you can use np.set_printoptions(precision=2).

Write a function normalise_data(array), which takes one NumPy array
as an argument.Hint:
make use of the array structure and Numpy math functions to complete the task.
"""

"""
Test 1:
import numpy as np
normalise_data(np.array([[1.0, 2.0], [4.0, 3.0]]))
Result 1:
[[-1.34 -0.45]
[ 1.34  0.45]]
Test 2:
import numpy as np
normalise_data(np.array([[0, 1], [1, 0]]))
Result 2:
[[-1  1]
[ 1 -1]]
Test 3:
import numpy as np
normalise_data(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]]))
Result:
[[-1.22  0.    1.22]
 [ 0.    1.22 -1.22]
 [ 1.22 -1.22  0.  ]]
"""

import numpy as np


def normalise_data(array):
    one = array.mean(axis=0)
    two = array - one
    three = np.std(array)
    four = two / three
    np.set_printoptions(precision=2)
    #print(f'{four}', end=' ')
    #print(four, end=' ')
    if four is four.astype(float):
        five = four.dtype(int)
        print(four)
    else:
        print(four.astype(float))
