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
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')														###### HERE
# we could also have a placeholder variable for the class number, but we will instead calculate it using argmax.
# Note that this is a TensorFlow operator, so nothing is calculated at this point.
y_true_cls = tf.argmax(y_true, dimension=1)

# Create the first convolutional layer
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
# Create the second convolutional layer
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

# Flatten layer to feed into the fully connected layers
layer_flat, num_features = flatten_layer(layer_conv2)

# Add a fully connected layer to the network. The input is the flattened layer from the previous convolution.
# The number of nodes in the fully connected layer is fc_size. ReLU is used so we can learn non-linear relations
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

# Add the last fully connected layer that outputs vectors of length 10 for determining which of the 10 classes the input image belongs to.
# Note that ReLU is not used in this layer
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

# The second fully-connected layer estimates how likely it is that the input image belongs to each of the 10 classes.
# However, these estimates are a bit rough and difficult to interpret because the numbers may be very small of large,
# so we want to normalize them so that each element is limited between zero and one and the 10 elements sum to one.
# This is calculated using the softmax function and the result is stored in y_pred
y_pred = tf.nn.softmax(layer_fc2)

# The class number is the index of the largest element
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Cost function to be optimized
# TensorFlow has a built function for calculating cross-entropy. 
# Note that the function calculates the softmax internally so we must use the output of layer_fc2 directly 
# rather than y_pred which has already had the softmax applied.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
# In order to use the cross-entropy to guide the optimization of the model's variables we need a single scalar value,
# so we simply take the average of the cross-entropy for all the image clasifications
cost = tf.reduce_mean(cross_entropy)

# Define the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Define performance measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# This calculates the classification accuracy by first type-casting the vector of booleans to floats, so that False becomes 0
# and True becomes 1, and then calculating the average of these numbers
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorFlow Run
session = tf.Session()

# Initialize variables
session.run(tf.global_variables_initializer())

# Helper function to perform optimization iterations
train_batch_size = 64

total_iterations = 0
def optimize(num_iterations):
	global total_iterations
	start_time = time.time()

	for i in range(total_iterations, total_iterations + num_iterations):
		# Get a batch of training examples
		# x_batch now holds a batch of images and y_true_batch are the true labels for those imagse
		x_batch, y_true_batch = data.train.next_batch(train_batch_size)

		# Put the batch into a dict with propre names for placeholder variables in the TensorFlow graph.
		feed_dict_train = {x: x_batch, y_true:y_true_batch}

		# TensorFlow assigns the variables in feed_dict_train to the placeholder variables and then runs the optimizer.
		session.run(optimizer, feed_dict=feed_dict_train)

		# Print status every 100 iterations
		if i % 100 == 0:
			acc = session.run(accuracy, feed_dict=feed_dict_train)
			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
			print(msg.format(i + 1, acc))

	# Update the total number of iterations performed
	total_iterations += num_iterations

	# Ending time
	end_time = time.time()
	time_diff = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))

# Helper function to plot example errors
def plot_example_errors(cls_pred, correct):
	incorrect = (correct == False)
	images = data.test.images[incorrect]
	cls_pred = cls_pred[incorrect]
	cls_true = data.test.cls[incorrect]
	plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

# Helper function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
	cls_true = data.test.cls
	cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
	print(cm)
	plt.matshow(cm)
	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks, range(num_classes))
	plt.yticks(tick_marks, range(num_classes))
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()

# Helper function for showing the performance
# Split the test set into smaller batches of this size
test_batch_size = 256
def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
	# number of images in the test set
	num_test = len(data.test.images)
	# Allocate an array for the predicted classes which will be calculated in batches and filled into this array
	cls_pred = np.zeros(shape=num_test, dtype=np.int)

	# Iterate through all the batches
	i = 0

	while i < num_test:
		# The ending index for the next batch is denoted j
		j = min(i + test_batch_size, num_test)

		# Get the images from the test-set between index i and j
		images = data.test.images[i:j, :]

		# Get the associated labels
		labels = data.test.labels[i:j, :]

		# Create a feed-dict with these images and labels
		feed_dict = {x:images, y_true: labels}
		cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

		# Set the start index of the next batch to the end index of the current batch
		i = j

	cls_true = data.test.cls
	# Create a boolean array whether each image is corectly classified
	correct = (cls_true == cls_pred)

	# Calculate the number of correctly classified images
	correct_sum = correct.sum()

	# Classification accuracy
	acc = float(correct_sum)/num_test

	msg = "Accuracy on test-set: {0:.1%} ({1} / {2})"
	print(msg.format(acc, correct_sum, num_test))

	# Plot some mis-classifications, if desired
	if show_example_errors:
		print("Example errors: ")
		plot_example_errors(cls_pred=cls_pred, correct=correct)

	# Plot the confusion matrix, if desired
	if show_confusion_matrix:
		print("Confusion matrix: ")
		plot_confusion_matrix(cls_pred=cls_pred)

# Helper function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
	# Assume weights are TensorFlow ops for 4-dim variables
	# eg weights_conv1 or weights_conv2
	# Retreive the values of the weight-variables from TensorFlow
	# A feed-dict is not necessary because nothing is calculated
	w = session.run(weights)
	w_min = np.min(w)
	w_max = np.max(w)

	# Number of filters used in the convolutional layer
	num_filters = w.shape[3]
	num_grids = math.ceil(math.sqrt(num_filters))

	# Create figure with a gris of subplots
	fig, axes = plt.subplots(num_grids, num_grids)

	# Plot all the filter weights
	for i, ax in enumerate(axes.flat):
		# Only plot the valid filter weights
		if i < num_filters:
			img = w[:, input_channel, i]
			ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()

# Helper function for plotting the output of a convolutional layer
def plot_conv_layer(layer, image):
	# Create a feed-dict containing only one image
	# We do not need to feed y_true because it is not used in this calculation
	feed_dict = {x: [image]}
	values = session.run(layer, feed_dict=feed_dict)
	num_filters = values.shape[3]
	num_grids = math.ceil(math.sqrt(num_filters))
	fig, axes = plt.subplots(num_grids, num_grids)

	# Plot the output images of all the filters
	for i, ax in enumerate(axes.flat):
		if i < num_filters:
			# Get the output image using the i'th filter
			# See new_conv_layer() for details on the format of this 4-dim tensor
			img = values[0, :, :, i]

			# Plot image
			ax.imshow(img, interpolation='nearest', cmap='binary')

		# Remove ticks from the plot
		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()

optimize(num_iterations=1000)
print_test_accuracy(True, True)
# 98.8% accuracy on test set with 10000 iterations