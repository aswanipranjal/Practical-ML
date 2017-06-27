import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import math

# Helper function to plot images
def plot_images(images, cls_true, cls_pred=None, img_shape=None):
	assert len(images) == len(cls_true) == 9

	# Create figure with 2x2 subplots
	fig, axes = plt.subplots(3, 3)
	fig_subplots_adjust(hspace=0.3, wspace=0.3)
	for i, ax in enumerate(axes.flat):
		ax.imshow(images[i].reshape(img_shape), cmap='binary')
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

		ax.set_xlabel(xlabel)
		# Remove ticks from plot
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

# Function to define a new convolutional layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
	# Shape of the filter weights according to the tensorflow API
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	# New weights for the fiters
	weights = new_weights(shape=shape)
	# New biases, one for each filter
	biases = new _biases(length=num_filters)
	# Create tensorflow  operation for convolution
	layer = tf.nn.conv2d(input=input, fliter=weights, strides=[1, 1, 1, 1], padding='SAME')
	# Add biases
	layer += biases

	if use_pooling:
		# If it is a 2x2 max pooling, from 2x2 moving window, take the largest value, then move 2 pixels to the right
		layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	layer = tf.nn.relu(layer)
	return layer, weights

# Helper function for flattening the layer
def flatten_layer(layer):
	layer_shape = layer.get_shape()
	# Assumed layer shape = [n_images, img_height, img_width, n_chanels]
	# The number of features is img_height * img_width * n_channels
	num_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat, num_features

# Helper function to create a fully connected layer
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	# Create new weights and biases
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)
	layer = tf.matmul(input, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)

	return layer

# Helper function to plot exmaple errors
def plot_example_errors(cls_pred, correct, data, img_shape):
	incorrect = (correct == False)
	images = data.test.images[incorrect]
	cls_pred = cls_pred[incorrect]
	cls_true = data.test.cls[incorrect]
	