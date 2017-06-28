import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import time
import math
import os
import prettytensor as pt
from tensorflow.examples.tutorials.mnist import input_data

print('TensorFlow version: {}', tf.__version__)
print('PrettyTensor version: {}', pt.__version__)
data = input_data.read_data_sets("tmp/data", one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)
print(combined_images.shape)
print(combined_labels.shape)
combined_size = len(combined_images)
print(combined_size)
train_size = int(0.8 * combined_size)
validation_size = combined_size - train_size

# Helper function for splitting the data into random training and validation sets
def random_training_set():
	# Create a randomized index into the full/combined dataset
	idx = np.random.permutation(combined_size)
	# Split the random index into  training anf validation sets
	idx_train = idx[0:train_size]
	idx_validation = idx[train_size:]
	x_train = combined_images[idx_train, :]
	y_train = combined_labels[idx_train, :]
	x_validation = combined_images[idx_validation, :]
	y_validation = combined_labels[idx_validaiton, :]
	return x_train, y_train, x_validation, y_validation

# Data dimensions
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

# Helper function for plotting images
def plot_images(images, cls_true, ensemble_cls_pred=None, best_cls_pred=None):
	assert len(images) == len(cls_true)
	fig, axes = plt.subplots(3, 3)
	if ensemble_cls_pred is None:
		hspace = 0.3
	else:
		hspace = 1.0
	fig.subplots_adjust(hspace=hspace, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# There may not be enough images for all sub-plots
		if i < len(images):
			# Plot image
			ax.imshow(images[i].reshape(img_shape), cmap='binary')
			if ensemble_cls_pred is None:
				xlabel = "True: {0}".format(cls_true[i])
			else:
				msg = "True: {0}\nEnsemble: {1}\nBest net: {2}"
				xlabel = msg.format(cls_true[i], ensemble_cls_pred[i], best_cls_pred[i])
			# Show the classes as the labels on the x-axis
			ax.set_xlabel(xlabel)
		# Remove ticks from the plot
		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true=cls_true)

# TensorFLow graph
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.placeholder(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# PrettyTensor might not work with this data yet. Use tensorflow convolutional neural network primitives instead
x_pretty = pt.wrap(x_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
	y_pred, loss = x_pretty.conv2d(kernel=5, depth=16, name='layer_conv1').max_pool(kernel=2, stride=2).conv2d(kernel=5, depth=36, name='layer_conv2').max_pool(kernel=2, stride=2).flatten().fully_connected(size=128, name='layer_fc1').softmax_classifier(num_classes=num_classes, labels=y_true)
	