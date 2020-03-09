"""Write the contents for the function both_odd(), which gets two arguments
n1 and n2. We can assume that the arguments are positive integers.
The function should print "both odd", if both the argument n1 is odd and the
argument n2 is odd. Otherwise, the function should print "no".
Complete the function by adding code to the function body."""

def both_odd(n1, n2):
    if n1 % 2 != 0 and n2 % 2 != 0:
        print('both odd')
    else:
        print('no')
