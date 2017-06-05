import numpy as numpy
from sklearn import proprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('cancer_dataset.txt')
# most algorithms recognize -99999 as an outlier and will treat it as so
df.replace('?', -99999, inplace=True)
# KNN handles outliers very badly, so we need to be careful
df.drop(['id'], 1, inplace=True)
# to drop entries with missing data, we write df.dropna(inplace=True)

# df.drop(['_attribute_'], 1) basically returns a dataframe without the attribute column in it and can be converted into an array by numpy
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# the line below separates training and testing data into four arrays according to the percentage of data we specify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)