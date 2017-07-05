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

# getting the layer outputs
def get_layer_output(layer_name):
	# this assumes you are using ReLU as the activation function
	tensor_name = "network/" + layer_name + "/Relu:0"
	# Get the tensor with this name
	tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
	return tensor

output_conv1 = get_layer_output(layer_name='layer_conv1')
output_conv2 = get_layer_output(layer_name='layer_conv2')

# TensorFlow run
session = tf.Session()

save_dir = 'C:\\Users\\Aman Deep Singh\\Documents\\Python\\Practical ML\\convnet_cifar10_checkpoints\\'
if not os.path.exists(save_dir):
	print("Checkpoint directory not found. Press Ctrl + C quick because this program doesn\'t handle errors")

save_path = os.path.join(save_dir, 'cifar10_cnn')

# First try to restore the saved checkpoint. This may fail and arise an exception
try:
	print("Trying to restore last checkpoint...")
	last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
	saver.restore(session, save_path=last_chk_path)
	print("Restored checkpoint from: ", last_chk_path)
except:
	print("Failed to restore checkpoint. Initializing variables instead.")
	session.run(tf.global_variables_initializer())

# Helper function to get a random training batch
train_batch_size = 64

def random_batch():
	# Number of images in the trainin set
	num_images = len(images_train)

	idx = np.random.choice(num_imagesm, size=train_batch_size, replace=False)

	x_batch = images_train[idx, :, :, :]
	y_batch = labels_train[idx, :]

	return x_batch, y_batch

# Helper function to perform optimization
def optimize(num_iterations):
	start_time = time.time()

	for i in range(num_iterations):
		x_batch, y_true_batch = random_batch()
		feed_dict_train = {x: x_batch, y_true: y_true_batch}
		# run the optimizer using this batch of training data.
		# TensorFlow assigns the variables in feed_dict_train
		# to the placeholder variables and then runs the optimizer
		# We also want to retrieve the global_step counter
		i_global, _ = session.run([global_step, optimizer], feed_dict=feed_dict_train)

		if(i_global % 100 == 0) or (i == num_iterations - 1):
			batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
			msg = "Global Step: {0:>6}, Training batch accuracy: {1:>6.1%}"
			print(msg.format(i_global, batch_acc))

		# Save a checkpoint to disk every 1000 iterations (and last)
		if(i_global % 1000 == 0) or (i == num_iterations - 1):
			saver.save(session, save_path=save_path, global_step=global_step)
			print("Saved checkpoint.")

	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Helper function to plot example errors
def plot_example_errors(cls_pred, correct):
	incorrect = (correct == False)
	images = images_test[incorrect]
	cls_pred = cls_pred[incorrect]
	cls_true = cls_test[incorrect]
	plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])