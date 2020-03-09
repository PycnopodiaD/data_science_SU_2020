"""In this exercise, you write a function to transform temperatures expressed
in degrees Fahrenheit to degrees Celsius.
The temperatures can be represented in degrees Fahrenheit or degrees Celsius.
The boiling point of water is 100 C (in degrees Celsius) and 212 F
(in degrees Fahrenheit). The freezing point of water is 0 C
(in degrees Celsius) and 32 F (in degrees Fahrenheit). Solve for a linear
equation with the data above to transform the temperature given in degrees
Fahrenheit to degrees Celsius. Print the result in degrees Fahrenheit.
Print one decimal after the decimal point, such as 32.0 or 100.0."""

def fahrenheit_to_celsius(cels):
    fah = (cels - 32) * 5.0/9.0
    print(f'{fah:.1f} degrees Fahrenheit ')
