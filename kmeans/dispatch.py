import os
import sys

def dispatch_func(filename):

    with open(filename, 'rb') as f:
        for block in f:
            if b'\0' in block:
                os.system('python Jaccard_script.py')
                break
            else:
                os.system('python KMeans_script.py')
                break

if __name__ == '__main__':
    filename = sys.argv[1]
    dispatch_func(filename)
