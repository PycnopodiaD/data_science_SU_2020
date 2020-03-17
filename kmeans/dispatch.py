import os
import sys

def dispatch_func(data_path):
    check = np.loadtxt(data_path)
    if np.array_equal(check, check.astype(bool)) == True:
        runJac(data_path)
    else:
        runEluc(data_path)


if __name__ == '__main__':
    data_path = sys.argv[1]
    dispatch_func(data_path)
