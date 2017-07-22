import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import math

# Not plotting images as they are stored as numpy arrays
# def plot_images()

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(shape):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
	# Shape of the filter weights according to the tensorflow API
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	# New weights for filters
	weights = new_weights(shape=shape)
	# New biases, one for each filter
	biases = new_biases(length=num_filters)
	# Create tensorflow operation for convolution
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
	# Add biases
	layer += biases

	if use_pooling:
		# If it is a 2x2 max pooling, from 2x2 moving window, take the largest vaue and then move 2 pixels to the right
		layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	layer = tf.nn.relu(layer)
	return layer, weights

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	# Assumed layer shape = [n_images, img_height, img_width, n_channels]
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

def get_weights_variable(layer_name):
	with tf.variable_scope(layer_name, reuse=True):
		variable = tf.get_variable('weights')

	return variable

def predict_cls(images, labels, cls_true, x, y_true, session, y_pred_cls, batch_size=256):
	num_images =len(images)
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

def cls_accuracy(correct):
	correct_sum = correct.sum()
	acc = float(correct_sum) / len(correct)
	return acc, correct_sum
