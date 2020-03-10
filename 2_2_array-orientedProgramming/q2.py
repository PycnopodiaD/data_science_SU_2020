"""
Sometimes, data sets are weighted to set individual importance to variables or
individual cases. Variables are usually by convention the columns in a data
matrix, cases are the rows in a data matrix.

In this exercise, you will write a function to multiply a data matrix and its
columns by a weight column factor and its rows by a corresponding column
factor. The function weight_rows_columns(data, row_weight, column_weight)
takes three arguments: data set itself, which is of type NumPy array.
The variables row_weight and column_weight are vectors, which are represented
with Numpy arrays.

The function prints the resulting weighted matrix.
To control the precision in the function, use the command
"np.set_printoptions(precision=3)" within the function.
"""

"""
Test 1:
import numpy as np
data = np.array([[1,1,1], [2,2,2], [3,3,3]])
row_weight = np.array([3, 1.5, 1])
column_weight = np.array([1, 2, 3])
weight_rows_columns(data, row_weight, column_weight)
Result 1:
[[3. 6. 9.]
 [3. 6. 9.]
 [3. 6. 9.]]

Test 2:
import numpy as np
data = np.array([[1,1], [2,2], [3,3]])
row_weight = np.array([6, 3, 2])
column_weight = np.array([2, 2])
weight_rows_columns(data, row_weight, column_weight)
Result 2:
[[12 12]
 [12 12]
 [12 12]]
"""
"""
Test 3:
import numpy as np
data = np.array([[1,1], [2,2], [3,3]])
row_weight = np.array([6, 0, 0])
column_weight = np.array([1, 1])
weight_rows_columns(data, row_weight, column_weight)
Result 3:
[[6 6]
 [0 0]
 [0 0]]
"""
import numpy as np

def weight_rows_columns(data, row_weight, column_weight):
    #new_array = column_weight * row_weight[0]
    #new_array2 = np.multiply(np.ones((len(new_array),len(new_array))),new_array)
    #new_array2 = numpy.array([new_array for _ in xrange(n)])
    #new_array2 = np.vstack([new_array]*3)
    #new_data = (data * row_weight * column_weight).T
    #new_data2 = new_data
    dot = np.dot(row_weight[:,None],column_weight[None])
    dot2 = dot * data
    np.set_printoptions(precision=3)
    print(dot2, end=' ')
