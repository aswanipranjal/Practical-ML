import numpy as numpy
from sklearn import proprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('cancer_dataset.txt')
# most algorithms recognize -99999 as an outlier and will treat it as so
df.replace('?', -99999, inplace=True)
# KNN handles outliers very badly, so we need to be careful
df.drop(['id'], 1, inplace=True)