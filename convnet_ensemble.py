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
	y_validation = combined_labels[idx_validation, :]
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
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# PrettyTensor might not work with this data yet. Use tensorflow convolutional neural network primitives instead
# PrettyTensor gives predictions through the convolutional neural network it has built and also a loss measure that must be minimized
x_pretty = pt.wrap(x_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
	y_pred, loss = x_pretty.conv2d(kernel=5, depth=16, name='layer_conv1').max_pool(kernel=2, stride=2).conv2d(kernel=5, depth=36, name='layer_conv2').max_pool(kernel=2, stride=2).flatten().fully_connected(size=128, name='layer_fc1').softmax_classifier(num_classes=num_classes, labels=y_true)

# Optimization method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver(max_to_keep=100)
save_dir = 'C:\\Users\\Aman Deep Singh\\Documents\\Python\\Practical ML\\convnet_ensemble_checkpoints\\'
if not os.path.exists(save_dir):
	print("Save filepath does not exist")

def get_save_path(net_number):
	return save_dir + 'network' + str(net_number)

# TensorFlow session
session = tf.Session()
def init_variables():
	# Probably need to change this line
	session.run(tf.initialize_all_variables())

train_batch_size = 64
# Function for selectign a random training-set from the given training set size
def random_batch(x_train, y_train):
	num_images = len(x_train)
	idx = np.random.choice(num_images, size=train_batch_size, replace=False)
	x_batch = x_train[idx, :]
	y_batch = y_train[idx, :]
	return x_batch, y_batch

# Helper function to perform optimization iterations
def optimize(num_iterations, x_train, y_train):
	start_time = time.time()
	for i in range(num_iterations):
		x_batch, y_true_batch = random_batch(x_train, y_train)
		feed_dict_train = {x: x_batch, y_true: y_true_batch}
		session.run(optimizer, feed_dict=feed_dict_train)
		if i % 100 == 0:
			acc = session.run(accuracy, feed_dict=feed_dict_train)
			msg = "Optimization iteration: {0:>6}, Training batch accuracy: {1:>6.1%}"
			print(msg.format(i + 1, acc))
	end_time = time.time()
	time_diff = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))

# Create ensemble of neural networks
num_networks = 5
num_iterations = 1000

# Create the ensemble of neural networks. All netrworks use the same TensorFlow graph as defined above. The variables are all initialized randomly and then optimized. The values of the weights are saved to disk so that they can be reloaded later
if True:
	for i in range(num_networks):
		print("Neural network: {0}".format(i))
		x_train, y_train, _, _ = random_training_set()
		session.run(tf.global_variables_initializer())
		optimize(num_iterations=num_iterations, x_train=x_train, y_train=y_train)
		saver.save(sess=session, save_path=get_save_path(i))
		# To print a newline
		print()

# Helper functions
batch_size = 256
def predict_labels(images):
	num_images = len(images)
	pred_labels = np.zeros(shape=(num_images, num_classes), dtype=np.float)
	i = 0
	while i < num_images:
		j = min(i + batch_size, num_images)
		# Create a feed-dict with the images between i and j
		feed_dict = {x: images[i:j, :]}
		pred_labels[i:j] = session.run(y_pred, feed_dict=feed_dict)
		i = j
	return pred_labels

# Calculate boolean array hether the predicted classes for the images are correct
def correct_prediction(images, labels, cls_true):
	# Calculate the predicted labels
	pred_labels = predict_labels(images=images)
	# Calculate the predicted class-number for each image
	cls_pred = np.argmax(pred_labels, axis=1)
	# Create a boolean array whether each image is correctly
	correct = (cls_true == cls_pred)
	return correct

# Calculate a boolean array whether the images in the test-set are classified correctly
def test_correct():
	return correct_prediction(images=data.test.images, labels=data.test.labels, cls_true=data.test.cls)

# Calculate a boolean array whether the images in the validation set are classified correctly
def validation_correct():
	return correct_prediction(images=data.validation.images, labels=data.validation.images, cls_true=data.validation.cls)

def classification_accuracy(correct):
	return correct.mean()

def test_accuracy():
	correct = test_correct()
	return classification_accuracy(correct)

def ensemble_predictions():
	pred_labels = []
	test_accuracies = []
	val_accuracies = []
	for i in range(num_networks):
		saver.restore(sess=session, save_path=get_save_path(i))
		test_acc = test_accuracy()
		test_accuracies.append(test_acc)
		val_acc = validation_accuracy()
		val_accuracies.append(val_acc)
		msg = "Network: {0}, Accuracy on validation set: {1:.4f}, Test set: {2:.4f}"
		print(msg.format(i, val_acc, test_acc))
		pred = predict_labels(images=data.test.images)
		pred_labels.append(pred)

	return np.array(pred_labels), np.array(test_accuracies), np.array(val_accuracies)

pred_labels, test_accuracies, val_accuracies = ensemble_predictions()

print("Mean test accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test accuracy: {0:.4f}".format(np.min(test_accuracies)))
print("Max test accuracy: {0:.4f}".format(np.max(test_accuracies)))
ensemble_pred_labels = np.mean(pred_labels, axis=0)
print(ensemble_pred_labels.shape)
ensemble_correct = (ensemble_cls_pred == data.test.cls)
ensemble_incorrect = np.logical_not(ensemble_correct)

# Best neural network
# We now find the single neural network that performed best on the test set
best_net = np.argmax(test_accuracies)
test_accuracies[best_net]
best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
best_net_correct = (best_net_cls_pred == data.test.cls)
best_net_incorrect = np.logical_not(best_net_correct)

# Comparison of best net vs the best single network
np.sum(ensemble_correct)
np.sum(best_net_correct)
ensemble_better = np.logical_and(best_net_incorrect, ensemble_correct)
ensemble_better.sum()
best_net_better.sum()

def plot_images_comparison(idx):
	plot_images(images=data.test.images[idx, :], cls_true=data.test.cls[idx], ensemble_cls_pred=ensemble_cls_pred[idx], best_cls_pred=best_net_cls_pred[idx])

# Function for printing predicted labels
def print_labels(labels, idx, num=1):
	labels = labels[idx, :]
	labels = labels[0:num, :]
	labels_rounded = np.round(labels, 2)
	print(labels_rounded)

def print_labels_ensemble(idx, **kwargs):
	print_labels(labels=ensemble_pred_labels, idx=idx, **kwargs)

def print_labels_best_net(idx, **kwargs):
	print_labels(labels=best_net_pred_labels, idx=idx, **kwargs)

def print_labels_all_nets(idx):
	for i in range(num_networks):
		print_labels(labels=pred_labels[i, :, :], idx=idx, num=1)


plot_images_comparison(idx=ensemble_better)