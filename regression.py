import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# didn't quite understand the difference between features and labels
quandl.ApiConfig.api_key = 'VDxfKkzAm8MFL1fZsSat'
df = quandl.get_table('WIKI/PRICES')
# print(df.head())
df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
df['hl_pct'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100
df['pct_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100
# The ones below are features
df = df[['adj_close', 'hl_pct', 'pct_change', 'adj_volume']]
# print(df.head())

forecast_col = 'adj_close'
# fills specified value wherever 'fill' is not available
# In machine learning, we don't want to have null data. We could remove the 
# entire column, but that would lead to 'wastage' of data in the other attributes
# An outlandish value in place of the unavailable data is more or less regarded as an 
# outlying value and will be 'ignored'
# this is better than sacrificing data
df.fillna(-99999, inplace=True)

# here we are trying to predict out 10% of the dataframe
forecast_out = int(math.ceil(0.01*len(df)))

# we are basically creating a space ten% days out into the future, thus the negative shift
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)
print(df.head())

# features : X
# labels : y
# because X contains features, we are dropping the label column
# df.drop() returns a new dataframe which is being converted into an array by numpy
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# preprocesses and normalizes all data points together
X = preprocessing.scale(X)

df.dropna(inplace = True)
y = np.array(df['label'])

# print(len(X), len(y))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)