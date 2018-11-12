import itertools
import sys

def defineInput(size=6):
    lst = list(itertools.product([0, 1], repeat=size))
    return lst

def label(L):
    if len(L)%2 != 0:
        sys.exit("Invalid list size")
    first_value = 0
    power = len(L)//2 - 1
    for i in range(len(L)//2):
        first_value+=L[i]*2**(power-i)
    second_value = 0
    for i in range(len(L)//2,len(L)):
        second_value+=L[i]*2**(power-i+len(L)//2)
    return first_value>second_value

def getValuesAndLabels(size=6):
    vals = defineInput(size)
    labels = [label(vals[i]) for i in range(len(vals))]
    return vals, labels

if __name__ == "__main__":
    lst = defineInput()
    for elem in lst:
        print("label :", label(elem), "\nvalues: ", elem )
