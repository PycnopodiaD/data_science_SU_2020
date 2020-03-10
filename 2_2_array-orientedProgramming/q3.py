"""
In this exercise, you will use the basic functionality of Pandas DataFrames.
Pandas DataFrames are a structure, which can be seen as an extended,
or decorated Numpy array. Columns as well as rows can have names, also Pandas
library provides lots of functions to process and summarise DataFrames.

Write a function sum_of_positive_x(data), which takes a DataFrame as an
argument. The function selects the variable by the name "x" and sums all
the entries in that column, which are positive (>0). The function prints
the result. To control the precision of the output, use the
"pd.set_option('precision', 2)" within your function.
"""

"""
Test 1:
import numpy as np
import pandas as pd
my_frame = pd.DataFrame({'a': [1,2,3,4], 'x': [-1, 0, 7, 35], 'c': [3, 5, 5, 1]})
sum_of_positive_x(my_frame)

Result 1:
42

Test 2:
import numpy as np
import pandas as pd
my_frame = pd.DataFrame({'a': [1,2,3,4], 'b': [-1, 0, 7, 35], 'c': [1,2,3,4], 'x': [3, 5, 5, 1]})
sum_of_positive_x(my_frame)

Result 2:
14
"""

import numpy as np
import pandas as pd

def sum_of_positive_x(data):
    #df2 = my_frame[(my_frame > 0).all()]
    df2 = my_frame[my_frame < 0] = 0
    one = np.array(df2['x'])
    #two = np.absolute(one)
    pd.set_option('precision', 2)
    print(f'{one.sum()}')
