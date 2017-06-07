import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1: 'r', -1: 'b'}
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1, 1, 1)

	# train
	def fit(self, data):
		self.data = data
		# opt_dict = {||w||: [w, b]}
		opt_dict = {}
		transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

		all_data = []
		for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					all_data.append(feature)

		self.max_feature_value = max(all_data)
		self.min_feature_value = min(all_data)
		all_data = None

	def predict(self, features):
		# sign of (X.w + b)
		classification = np.sign(np.dot(np.array(features), self.w) + self.b)
		return classification

data_dict = {-1:np.array([[1, 7], [2, 8], [3, 8],]), 1:np.array([[5, 1], [6, -1], [7, 3]])}