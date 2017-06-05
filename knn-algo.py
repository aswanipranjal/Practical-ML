# implementation of K-Nearest-Neighbors algorithm
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
# we are importing warnings to warn the user when they are trying to input a nonsense number for K
import warnings
from collections import Counter
style.use('fivethirtyeight')

# example
# plot1 = [1, 3]
# plot2 = [2, 5]
# euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1] - plot2[1])**2)
# print(euclidean_distance)

# features that correspond to the class of 'k'
# class 'r' has labels
dataset = {'k':[[1, 2], [2, 3], [3, 1]], 'r':[[6, 5], [7, 7], [8, 6]]}
