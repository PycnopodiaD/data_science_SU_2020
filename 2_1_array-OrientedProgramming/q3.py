"""In this exercise, you will write a program to check whether a string is a palindrome.
Palindrome is a word or a sentence that spells the same forwards and backwards.
Some examples include: lol, radar, never odd or even.
Your program should not make a difference between uppercase and lowercase letters, that is, it should be case-insensitive.
Your solutions should ignore spaces, like in the example "Never odd or even", which is considered to be a palindrome.
Write a function that prints the result in the style: "lol is a palindrome" or "Hello is not a palindrome"."""

"""Test
is_palindrome('lol')
Result
lol is a palindrome"""

"""Test
is_palindrome('Hello, world')
Result
Hello, world is not a palindrome"""

"""Test
is_palindrome('Never odd or even')
Result
Never odd or even is a palindrome"""

def is_palindrome(my_string):
    # make it suitable for caseless comparison
    #my_string = my_string.casefold()
    #make the string lowercase
    low_str = my_string.lower()
    # reverse the string
    #rev_str = reversed(low_str.replace(' ', ''))
    rep_str = low_str.replace(' ', '')
    rev_str = rep_str[::-1]
# check if the string is equal to its reverse
    if rep_str == rev_str:
        print(f'{my_string} is a palindrome')
    else:
        print(f'{my_string} is not a palindrome')
