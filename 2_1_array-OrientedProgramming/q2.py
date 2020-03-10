"""In this exercise, you will calculate an error function between the known values of a function and the predicted values.
You will write a function that gets two parameters as lists.
The first list contains the known values of the function, which will be recorded in the data set.
The second list contains the predicted values of the function, created by your predictor function.
The function will calculate the means of the squared errors, an error measure also known as the MSE.

In order to calculate the MSE for all predictions, you will calculate the difference of the i'th
known value and the i'th predicted value, square the difference, and calculate the average over all the squared differences.

This measure will quantify how well, or accurately your prediction function is performing.
 The result is one number, which you will print out. The field width of the printed result is 4 characters, use two decimals.
 Check first that the lists have the same length, that is, there is the same number of known and predicted values.
 If this is not the case, print out "Error: parameter lists must have the same length"."""

Test
mean_squared_error([1.2, 2.4, 4.8], [0.8, 2.0, 4.4])
Result
MSE: 0.16

Test
mean_squared_error([1.2, 2.4, 4.8], [1.2, 2.4, 4.8])
Result
MSE: 0.00


def mean_squared_error(known_values, predicted_values):
    ##get the last value in each list
    if len(known_values) == len(predicted_values):
        together_vals = [((known_values[0] - predicted_values[0]) ** 2), ((known_values[1] - predicted_values[1]) ** 2), ((known_values[2] - predicted_values[2]) ** 2)]
    ##calculate the average over all the squared differences
    ##output value needs {:4.2f}
        print(f'MSE: {sum(together_vals)/len(together_vals):.2f}')
    else:
        print('Error: parameter lists must have the same length')
