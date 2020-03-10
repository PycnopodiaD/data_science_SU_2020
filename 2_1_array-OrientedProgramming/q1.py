"""In this exercise, you will calculate some basic descriptive statistics of lists of integer numbers.
You will write a function, which will get one parameter: list of integers.
The function will first make sure that all the items in the list are integers.
Then it will calculate basic statistics from the collection of integers and print out a following summary statement,
as demonstrated in the example answer below.
For the average, report one decimal place.
If the list does contain other types of numbers such as floating point numbers (or any type not integers),
print "Error: parameter list must contain integers"."""

"""Test
descriptive_statistics([1, 2, 4, 10, 2])"""
"""Result
number of items: 5
sum: 19
largest: 10
smallest: 1
average: 3.8"""

###

def descriptive_statistics(my_list):
#make sure all items are integers
    if all(type(item)==int for item in my_list):
#print number of items
        print(f'number of items: {len(my_list)}')
#print sum
        print(f'sum: {sum(my_list)}')
#print max
        print(f'largest: {max(my_list)}')
#print min
        print(f'smallest: {min(my_list)}')
#print average with 2 decimal places
        print(f'average: {sum(my_list)/len(my_list):.1f}')
    else:
        print('Error: parameter list must contain integers')

descriptive_statistics([1, 2, 4, 10, 2])
