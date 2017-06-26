import util as util
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
from datetime import timedelta
import os

# Constants
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

filter_size1 = 5
num_filters1 = 16
filter_size2 = 5
num_filters2 = 36
fc_size = 128
batch_size = 256
train_batch_size = 64
# Best validation accuracy seen so far
best_validation_accuracy = 0.0
# Iterations for last improvement to validation accuracy
last_improvement = 0
# Stop optimization if no improvement in this many iterations to avoid over-feeding
require_improvement = 1000
# Counter for total number of iterations so far
total_iterations = 0

# Class labels one-hot encoded
data.tst.cls = np.argmax(data.validation.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

# Define placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Function for new convolutional layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	weights = new_weights(shape=shape)
	biases = new_biases(length=num_filters)
	# tf.nn.conv2d requires the input to be 2-dimensional, and returns a 2-dimensional tensorflow layer object
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
	layer += biases
	if use_pooling:
		layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	layer = tf.nn.relu(layer)
	return layer, weights

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = np.array(layer_shape[1:4], dtype=int).prod()
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)
	layer = tf.matmul(input, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer

# function to define the convolutional neural network
def define_cnn(x_image=None, num_channels=None, filter_size1=None, num_filters1=None, filter_size2=None, num_filters2=None, fc_size=None, num_classes=None):
	# Convolutional layer 1
	layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
	print(layer_conv1.shape)

	# Convolutional layer 2
	layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
	print(layer_conv2.shape)

	layer_flat, num_features = flatten_layer(layer_conv2)
	print(layer_flat.shape)
	print(num_features) # 1764

	# Fully connected layer 1
	layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)
	print(layer_fc1.shape)

	# Fully connected layer 2
	layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)
	print(layer_fc2.shape)

	# Predicted class
	y_pred = tf.nn.softmax(layer_fc2)
	y_pred_cls = tf.nn.argmax(y_pred, dimension=1)

	# Cost function to be optimimzed
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true))

	return y_pred, y_pred_cls, loss, weights_conv1, weights_conv2