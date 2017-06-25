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

# Helper functions to create new variables
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

# A TensorFlow graph consists of the following
# Placeholder variables used for inputting data to the graph
# Variables that are going to be optimized so as to make the convolutional netork perform better
# The mathematical formulas for the convolutional network
# A cost measure used to guide the optimization of the variables
# An optimization method which updates the variables

# Helper function for creating a new convolutional layer
# Nothing is calculated here, we are just adding the mathematical formulas to the TensorFlow graph
# Assumption: The input is a 4-dim tensor with the following dimensions
# Image number
# Y-axis of each image
# X-axis of each image
# Channels of each image
# The output is anothe 4-dim tensor with the following dimensions
# Image number (same as input)
# Y-axis of each image. If 2x2 pooling is usedm then the width of the input images is divided by 2
# X-axis of each image. If 2x2 pooling is usedm then the width of the input images is divided by 2
# Channels produced by the convolutional filters
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
	# Shape of the filter weights for the convolution.
	# This format is determined by the TensorFlow API
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	# Create new weights with given shape
	weights = new_weights(shape=shape)
	# Create new biases, one for each filter
	biases = new_biases(length=num_filters)

	# Create TensorFlow operation for convolution
	# Note the strides are set to 1 in all dimensions
	# The first and last strindes must always be 1
	# The padding is set to 'SAME' which means the input image is padded with zeros so the size of the output is same
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
	# Add the biases to the results of the convolution. A bias value is added to each filter channel
	layer += biases
	# Use pooling to down-sample the resolution?
	if use_pooling:
		# This is 2x2 max-pooling, which means that we consider 2x2 windows and select the largest value in each window
		# Then we move 2 pixels to the next window
		layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	# Rectified Linear Unit (ReLU)
	# It calculates max(x, 0) for each input pixel x.
	# This adds some non-linearity to the formula and allows us to learn more complicated functions
	layer = tf.nn.relu(layer)
	# ReLU is normally execute before pooling, bu since ReLU(max_pool(x)) == max_pool(ReLU(x)) we can save 75% of the ReLU operations
	# by max-pooling first

	# We return both the resulting layer and the filter-weights because we will plot the weights later.
	return layer, weights

# The output of a convolutional layer is four-dimensional, and in order to use a fully-connected layer, we need 2-dimensional input
def flatten_layer(layer):
	layer_shape = layer.get_shape()

	# The shape of the layer is assumed to be 
	# layer_shape = [num_images, img_width, img_height, num_channels]
	num_features = np.array(layer_shape[1:4], dtype=int).prod()

	# Reshape the layer to [num_images, num_features]
	layer_flat = tf.reshape(layer, [-1, num_features])

	# The shape of the flattened layer is now:
	# [num_images, img_width * img_height * num_channels]
	return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)

	# Calculate the layer as the matrix multiplication of the input and weights, and then add the biases
	layer = tf.matmul(input, weights) + biases

	# Use ReLU?
	if use_relu:
		layer = tf.nn.relu(layer)

	return layer

# Placeholder variables
# Input
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
# The convolutional layer expects the inputs to be a four-dimensional tensor, so we reshape it
# Note that the similarity of the three values can be inferred automatically by using -1 for the size of the first dimension
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
# The shape of this placehilder variable is: [None, num_classes], because y_true is one-hot encoded
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
# we could also have a placeholder variable for the class number, but we will instead calculate it using argmax.
# Note that this is a TensorFlow operator, so nothing is calculated at this point.
y_true_cls = tf.argmax(y_true, dimension=1)

# Create the first convolutional layer
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
# Create the second convolutional layer
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

# Flatten layer to feed into the fully connected layers
flatten_layer, num_features = flatten_layer(layer_conv2)

# Add a fully connected layer to the network. The input is the flattened layer from the previous convolution.
# The number of nodes in the fully connected layer is fc_size. ReLU is used so we can learn non-linear relations
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

# Add the last fully connected layer that outputs vectors of length 10 for determining which of the 10 classes the input image belongs to.
# Note that ReLU is not used in this layer
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)
