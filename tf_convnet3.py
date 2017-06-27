# https://github.com/miga101/tf_mnist_cnn/blob/master/cnn_mnist_.py
import tf_convnet_utils as util
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
from datetime import timedelta
import os

img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

filter_size1 = 5
num_filters1=  16
filter_size2 = 5
num_filters2 = 36
fc_size = 128
batch_size = 256
train_batch_size = 64
best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 1000
total_iterations = 0

data = input_data.read_data_sets("tmp/data", one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

def define_cnn(x_image=None, num_channels=None, filter_size1=None, num_filters1=None, filter_size2=None, num_filters2=None, fc_size=None, num_classes=None):
	layer_conv1, weight_conv1 = util.new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
	print(layer_conv1.shape)

	layer_conv2, weight_conv2 = util.new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
	print(layer_conv2.shape)

	layer_flat, num_features = util.flatten_layer(layer_conv2)
	print(layer_flat.shape)
	