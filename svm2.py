# Siraj Raval's version of support vector machines
import numpy as np
import matlpotlib.pyplot as plt

def svm_sgd_plot(X, Y):
	# Initialize our SVMs weight vector with zeros (3 values)
	w = np.zeros(len(X[0]))
	# The learning rate
	eta = 1
	# How many iterations to train for
	epochs = 100000
	# Store misclassifications so we can plot how they change over time
	errors = []

	# training part, gradient descent part
	for epoch in range(1, epochs):
		error = 0
		for i, x in enumerate(X):
			# misclassification
			if (Y[i] * np.dot(X[i], w)) < 1:
				# misclassified: update the weights
				w = w + eta * ((X[i] * Y[i]) + (-2 * (1/epoch) * w))
				error = 1
			else:
				# correct classification, update our weights
				w = w + eta * (-2 * (1/epoch) * w)
		errors.append(error)

	# lets plot the rate of classification errors during training
	plt.plot(errors, '|')
	plt.ylim(0.5, 1.5)
	plt.axes().set_yticklabels([])
	plt.xlabel('Epoch')
	plt.ylabel('Misclassified')
	plt.show()

	return w