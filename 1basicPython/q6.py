"""Extend the previous exercise to handle conversion from both degrees Celsius
to degrees Fahrenheit and degrees Fahrenheit to degrees Celsius.
To which direction the conversion is done is determined by the first argument
to the function. If the argument is "f2c", the conversion is from Fahrenheit to
 Celsius, if the argument is "c2f", the conversion is from Celsius to
 Fahrenheit. You can make use of the previous exrecise, but you should also
 solve the linear equation to transform temperatures expressed in degrees
 Celsius to degrees Fahrenheit.
For example, the function call temperature_conversion('f2c', 451) is supposed
to transform 451 F to degrees Celsius. Similarly, the function call
temperature_conversion('c2f', 100) is supposed to transform the 100 C to
degrees Fahrenheit. Print the resulting temperature inside the function.
Print one decimal after the floating point (like 100.0 or 32.0)."""

def temperature_conversion(transform, temperature):
    if transform == 'f2c':
        fah = (temperature - 32) / 1.8
        print(f'{fah:.1f}')
    elif transform == 'c2f':
        cel = 9.0/5.0 * temperature + 32
        print(f'{cel:.1f}')
