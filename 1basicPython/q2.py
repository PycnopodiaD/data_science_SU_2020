"""Write the contents for the function even_or_odd_check(), which gets one
 argument n. You can make use of the previous exercise that you have completed.
  In this exercise, you must check that the argument n is positive
  integer (n > 0). If it is not, you should print "error" and exit.
  Otherwise, you should print "odd" or "even" according to the argument,
  like in the previous exercise.
Complete the function by adding code to the function body. """

def even_or_odd_check(n):
        if n > 0:
            print('error')
        elif n % 2 == 0:
            print('even')
        else:
            print('odd')




def even_or_odd(*n):
        n = int(input('Enter an integer: '))
        if n % 2 == 0:
            print(f'{n} is even')
        else:
            print(f'{n} is odd')

##val = int(input('Enter an integer: '))
