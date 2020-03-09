""" You will write a function that will print out a table of numbers consisting
 of integers from 1 to the function argument n. The first column output has
 this number. The columns are separated by the tab character (\t).
 The second column is the the first number squared (n * n).
 You will print as many rows as the argument tells you to. See the examples
 on the expected outputs."""

def print_squares(n):
    for squ in range(1,n+1):
        print(f'{squ}\t{squ ** 2}')
