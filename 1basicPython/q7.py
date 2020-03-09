"""In this exercise, you will prompt the user to enter his or her age, you
should print a question "Enter you age: " and then receive the age from the
terminal input. The inputted age will be a string, and you need to transform
it to a number to perform calculations. When you have transformed it to a
number, you can calculate how many years you will be in the working life until
retirement. Assuming that you would retire at 65, calculate and print out the
number of years that you will be still be working. For instance, if the age is
50, you should print "Still working for 15 years"
Check that the age is between 0 and 65. If the age is above 65, you can
print "You are already retired". If the age is below 0, for instance -5,
print out "See you again in 5 years!"."""

def years_until_retirement(*age):
    age = int(input('Enter your age: '))
    if age < 0:
        response = 0 - age
        print(f'See you again in {response} years!')
    elif age >= 0 and age <= 65:
        response = 65 - age
        print(f'Still working for {response} years')
    else:
        print('You are already retired')
