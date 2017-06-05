import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

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
forecast_out = int(math.ceil(0.001*len(df)))
print(forecast_out)

# we are basically creating a space ten% days out into the future, thus the negative shift
df['label'] = df[forecast_col].shift(-forecast_out)

# features : X
# labels : y
# because X contains features, we are dropping the label column
# df.drop() returns a new dataframe which is being converted into an array by numpy
X = np.array(df.drop(['label'], 1))
# preprocesses and normalizes all data points together
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)
y = np.array(df['label'])

# print(len(X), len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# clf = svm.SVR()
# clf = svm.SVR(kernel='poly')
# 
# The classifier below runs 10 threads at once
# clf = LinearRegression(n_jobs=10)
# 
# The classifier below runs maximum possible threads at once
# clf = LinearRegression(n_jobs=-1)
clf = LinearRegression()
clf.fit(X_train, y_train)
# at this stage, the classifier has been trained and we want to save (pickle) it
with open('linearregression.pickle', 'wb') as f:
	pickle.dump(clf, f) # dumps the classifier into the file f

# to open the pickle
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in);
# the above line loads the pickle
accuracy = clf.score(X_test, y_test)

# print(accuracy)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['adj_close'].plot()
df['forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()