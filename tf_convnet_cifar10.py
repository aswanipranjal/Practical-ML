# A convolutional neural network for classifying images in the CIFAR-10 dataset. It also shows how to use different
# networks during training and testing
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt

print("TensorFlow version: {}".format(tf.__version__))
print("PrettyTensor version: {}".format(pt.__version__))

import cifar10
# A 163 MB download
cifar10.data_path = "tmp/data/CIFAR-10/"
cifar10.maybe_download_and_extract()

class_names = cifar10.load_class_names()
print(class_names)

# Load the training set
images_train, cls_train, labels_train = cifar10.load_training_data()

# Load the test set
images_test, cls_test, labels_test = cifar10.load_test_data()
print("Size of: ")
print("-Training-set:\t\t{}".format(len(images_train)))
print("-Test-set:\t\t{}".format(len(images_test)))

from cifar10 import img_size, num_channels, num_classes
img_size_cropped = 24

def plot_images(images, cls_true, cls_pred=None, smooth=True):
	assert len(images) == len(cls_true) == 9
	fig, axes = pls.subplots(3, 3)
	if cls_pred is None:
		hspace = 0.3
	else:
		hspace = 0.6
	fig.subplots_adjust(hspace=hspace, wspace=0.3)
	for i, ax in enumerate(axes.flat):
		if smooth:
			interpolation = 'spline16'
		else:
			interpolation = 'nearest'
		ax.imshow(images[i, :, :, :], interpolation=interpolation)
		# Name of the true class
		cls_true_name = class_names[cls_true[i]]

		# Show true and predicted classes
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true_name)
		else:
			cls_pred_name = class_names[cls_pred[i]]
			xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
		ax.set_xlabel(xlabel)
		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()

# Plot a few images to see if it is correct
images = images_test[0:9]
cls_true = cls_test[0:9]
plot_images(images=images, cls_true=cls_true, smooth=False)
plot_images(images=images, cls_true=cls_true, smooth=True)

# Define TensorFlow graph
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_channels], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Preprocessing the image for feeding into the convolutional neural network
def pre_process_image(image, training):
	if training:
		# Randomly crop the input image
		image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
		# Randomly flip the image horizontally
		image = tf.image.random_flip_left_right(image)
		# Randomly adjust hue, contrast and saturation
		image = tf.image.random_hue(image, max_delta=0.05)
		image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
		image = tf.image.random_brightness(image, max_delta=0.2)
		image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

		# Limit the image pixels between [0, 1] in case of overflow
		image = tf.minimum(image, 1.0)
		image = tf.maximum(image, 0.0)
	else:
		# Crop the input image around the center so it is the same size as the images that are randomly cropped during training
		image = tf.image.resize_image_with_crop_or_pad(image, target_height=img_size_cropped, target_width=img_size_cropped)

	return image

def pre_process(images, training):
	# Use tensorflow to loop over all the input images and call the function above which takes a single image as input
	images = tf.map_fn(lambda image: pre_process_image(image, training), images)
	return images

# In order to plot the distorted images, we create the pre-processsing graph for tensorflow so that we may execute it later
distorted_images = pre_process(images=x, training=True)

# Helper function for creating main processing
def main_network(images, training):
	x_pretty = pt.wrap(images)

	# Pretty tensor uses special numbers to distinguish between the training and the testing phases
	if training:
		phase = pt.Phase.train
	else:
		phase = pt.Phase.infer

	# Create the convolutional neural network using pretty tansor and batch-normalization in the first layer
	with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
		y_pred, loss = x_pretty.conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).max_pool(kernel=2, stride=2).conv2d(kernel=5, depth=64, name='layer_conv2').max_pool(kernel=2, stride=2).flatten().fully_connected(size=256, name='layer_fc1').fully_connected(size=128, name='layer_fc2').softmax_classifier(num_classes=num_classes, labels=y_true)

	return y_pred, loss

# Helper function for creating a neural network instance
def create_network(training):
	# wrap the neural network in the scope named 'network'
	# create new variables during training, and reuse during testing
	with tf.variable_scope('network', reuse=not training):
		# Just rename the input placeholder variable for convenience
		images = x
		images = pre_process(images=images, training=training)
		y_pred, loss = main_network(images=images, training=training)

	return y_pred, loss

# Create neural network for training phase
# trainable=False means thath tensorflow will not try to optimize that variable
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
_, loss = create_network(training=True)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

# Create neural network for test phase/inference
y_pred, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define saver object
saver = tf.train.Saver()

# Get weights to plot (as they were indirectly created by prettytensor)
def get_weights_variable(layer_name):
	with tf.variable_scope("network/" + layer_name, reuse=True):
		variable = tf.get_variable('weights')
	return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')