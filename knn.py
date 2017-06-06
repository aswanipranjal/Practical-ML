import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

accuracies = []
for i in range(25):
	df = pd.read_csv('C:\\Users\\Aman Deep Singh\\Documents\\Python\\Practical ML\\cancer_dataset.txt')
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

	clf = neighbors.KNeighborsClassifier()
	clf.fit(X_train, y_train)

	# accuracy is different from confidence
	accuracy = clf.score(X_test, y_test)
	# print(accuracy)

	# # test data (eyeballs expect it to be benign)
	# example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
	# # DeprecationWarning: Passing 1D arrays as data is deprecated in sklearn0.17 and will raise ValueError in sklearn0.19. 
	# # Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample
	# # ^This error can be fixed by the line below
	# example_measures = example_measures.reshape(1, -1)

	# # For example, 2 predictions can be done like so:
	# # example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
	# # example_measures = example_measures.reshape(2, -1)

	# # _any_ number of predictions can be done like so:
	# # example_measures = np.array(__something__)
	# # example_measures = example_measures.reshape(len(example_measures), -1)

	# # We reshape numpy arrays so that we can feed it through scikit-learn

	# prediction = clf.predict(example_measures)
	# print(prediction)
	accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))