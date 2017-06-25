# Convolutional neural networks work by miving small filters across the input image. 
# This means the filters are re-used for recognizing patterns throughout the entire input image. 
# This makes the convolutional networks much more powerful than fully-connected networks with the same number of variables.
# This in turn makes ConvNets faster to train
# Filters are run over the input image, then it can be padded with zeros (white pixels). This causes the output image to be of
# the exact same dimension as the input image
# Furthermore, the output of the convolution may be passed through a so-called Rectified Linear Unit (ReLU) 
# (a low cost activation function), which merely ensures that the output is positive because negative values are set to zero.
# The output may also be down-sampled by so-called max-pooling, which considers small windows of 2x2 pixels and only keeps the largest
# of those pixels. This halves the resolution of the input image.
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# Configuration of Convolutional Neural Network
# Convolutional layer 1
filter_size1 = 5 # 5x5 pixels
num_filters1 = 16 # 16 of these filters

# Convolutional layer 2
filter_size2 = 5 # 5x5 pixels
num_filters2 = 36 # 36 of these filters

# Fully-connected layer
fc_size = 128 # Number of neurons in fully-connected layer

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("tmp/data", one_hot=True)

# The labels are one-hot encoded
data.test.cls= np.argmax(data.test.labels, axis=1)

# Data dimensions
img_size = 28
img_size_flat = img_size * img_size
# Tuple wih height and width of the arrays to reshape them
img_shape = (img_size, img_size)
# 1 for grayscale
num_channels = 1
num_classes = 10

# Helper function to plot images
def plot_images(images, cls_true, cls_pred=None):
	assert len(images) == len(cls_true) == 9

	# Create figure with 3x3 subplots
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Plot image
		ax.imshow(images[i].reshape(img_shape), cmap='binary')

		# Show true and predicted classes
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

		ax.set_xlabel(xlabel)

		# Remove ticks from the plot
		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()