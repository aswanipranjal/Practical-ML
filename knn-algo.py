# implementation of K-Nearest-Neighbors algorithm
from math import sqrt
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# we are importing warnings to warn the user when they are trying to input a nonsense number for K
import warnings
from collections import Counter
import pandas as pd
import random
# style.use('fivethirtyeight')

# example
# plot1 = [1, 3]
# plot2 = [2, 5]
# euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1] - plot2[1])**2)
# print(euclidean_distance)

# features that correspond to the class of 'k'
# class 'r' has labels
# dataset = {'k':[[1, 2], [2, 3], [3, 1]], 'r':[[6, 5], [7, 7], [8, 6]]}
# new_features = [5, 7]

# i corresponds to 'k' and 'r'
# ii corresponds to each feature (a pair of coordinates)

# The nested for loop below can be written as the one-liner even below
# for i in dataset:
	# for ii in dataset[i]:
		# plt.scatter(ii[0], ii[1], s=100, color=i)

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1])
# plt.show()

def KNN (data, predict, K=3):
	if len(data) >= K:
		warnings.warn('K is set to a value less than total voting groups')

	distances = []
	# group refers to 'k' or 'r'
	for group in data:
		for features in data[group]:
			# we will not do this:
			# euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1] - plot2[1])**2)
			# instead:
			# euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
			# this is faster:
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidean_distance, group])

	# i returns ~kind of a dictionary entry. i[1] returns the second element of that 'pair' probably
	# we sort distances in ascending order and take the first k entries into a new list called 'votes'
	# print(sorted(distances)) # Debug line
	votes = [i[1] for i in sorted(distances)[:K]]
	# print(votes) # Debug line
	# print(Counter(votes).most_common(1)) # Debug line
	# This line will print out the first entry of the first tuple in most_common 1 instance
	vote_result = Counter(votes).most_common(1)[0][0]

	return vote_result

# result = KNN(dataset, new_features, K=3)
# print(result)

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], color=result)
# plt.show()

df = pd.read_csv('C:\\Users\\Aman Deep Singh\\Documents\\Python\\Practical ML\\cancer_dataset.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# print(df.head())
# To ensure that everything in the dataframe is an int or a float, we convert it using a pandas prebuilt function
# (Sometimes there are quotes in the data which makes it a string)
full_data = df.astype(float).values.tolist()
# print(full_data[:10])

random.shuffle(full_data)

# Our version of train_test_split
test_size = 0.2;
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

# populating dictionaries
for i in train_data:
	# i[-1] accesses the last element, i.e. to find whether it is benign or malignant
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])

