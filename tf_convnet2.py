# A version of the convolutional neural network that will just train a network
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math

# Convolutional layer 1
filter_size1 = 5
num_filters1 = 16

# Convolutional layer 2
filter_size2 = 5
num_filters2 = 36

# Fully connected layer
fc_size = 128

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("tmp/data", one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)

# Data dimensions
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	weights = new_weights(shape=shape)
	biases = new_biases(length=num_filters)
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
	layer += biases
	if use_pooling:
		layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	layer = tf.nn.relu(layer)
	return layer, weights