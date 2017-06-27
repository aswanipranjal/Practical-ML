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
	plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9], img_shape=img_shape)
	plt.show()

def plot_confusion_matrix(cls_pred, data, num_classes):
	cls_true = data.test.cls
	cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
	print(cm)
	plt.matshow(cm)
	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks, range(num_classes))
	plt.yticks(tick_marks, range(num_classes))
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()

def get_weights_variable(layer_name):
	with tf.variable_scope(layer_name, reuse=True):
		variable = tf.get_variable('weights')

	return variable

def predict_cls(images, labels, cls_true, x, y_true, session, y_pred_cls, batch_size=256):
	num_images = len(images)
	cls_pred = np.zeros(shape=num_images, dtype=np.int)
	while i < num_images:
		j = min(i + batch_size, num_images)
		feed_dict = {x: images[i:j, :], y_true: labels[i:j, :]}
		cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
		i = j
	correct = (cls_true == cls_pred)
	return correct, cls_pred

def init_variables(session):
	session.run(tf.global_variables_initializer())

def predict_cls_test(data):
	return predict_cls(images=data.test.images, labels=data.test.labels, cls_true=data.test.cls)

def predict_cls_validation(data):
	return predict_cls(images=data.validation.images, labels=data.validation.labels, cls_true=data.validation.cls)

def cls_accuracy(correct):
	correct_sum = correct.sum()
	acc = float(correct_sum) / len(correct)
	return acc, correct_sum

def validation_accuracy():
	correct, _ = predict_cls_validation()
	return cls_accuracy(correct)

def plot_conv_weights(weights, input_channel=0, session=None):
	w = session.run(weights)
	print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
	w_min = np.min(w)
	w_max = np.max(w)
	num_filters = w.shape[3]
	num_grids = math.ceil(math.sqrt(num_filters))
	fig, axes = plt.subplots(num_grids, num_grids)
	for i, ax in enumerate(axes.flat):
		if i < num_filters:
			img = w[:, :, input_channel, i]
			ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()