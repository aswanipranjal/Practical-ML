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
import tflearn
from tflearn.nn.core import input_data, regression, local_response_normalization
from tflearn.nn.conv import conv_2d, max_pool_2d 
from tflearn.layers.conv import fully_connected

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
def process_image(image, training):
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