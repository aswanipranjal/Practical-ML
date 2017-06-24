import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("tmp/data", one_hot=True)
print("Size of: ")
print("-Training set: \t\t{}".format(len(data.train.labels)))
print("-Test set: \t\t{}".format(len(data.test.labels)))
print("-Validation set: \t{}".format(len(data.validation.labels)))
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10

data.test.cls = np.array([label.argmax() for label in data.test.labels])

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

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true=cls_true)

# First we define the placeholder variable for the input images. This allows us to change the images that are input to the
# TensorFlow graph. This is a so-called tensor, which just means that it is a multi-dimensional vector or matrix.
# The data type is set to float32 and the shape is set to [None, img_size_flat], where None means that the tensor 
# may hold an arbitrary number of images with each image being a vector of length img_size_flat
x = tf.placeholder(tf.float32, [None, img_size_flat])
# Next, we have the placeholder variable for the true labels associated with the images that were input in the placeholder
# variable x. The shape of this placeholder variable is [None, num_classes] which means it may hold an arbitrary number of labels
# and each label is a vector of length num_classes which is 10 in this case.
y_true = tf.placeholder(tf.float32, [None, num_classes])
# Finally we have the placeholder variable for the true class of each image in the placeholder variable x. These are integers
# and the dimensionality of this placeholder variable is set to [None] which means the placeholder variable is a one-dimensional
# vector of arbitrary length.
y_true_cls = tf.placeholder(tf.int64, [None])

weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

# Model
logits = tf.matmul(x, weights) + biases
# logits is a matrix with num_images rows and num_classes columns, where the element of the ith row and jth column is an estimate
# of how likely the ith input image is to be of the jth class
# however these estimates are a bit rough and difficult to interpret because the numbers may be very small or large, so we want 
# to normalize them so that each row of the logits matrix sums to one, and each element is limited between zero and one. This is
# calculated using the so-called softmax function and the result is stored in y_pred
y_pred = tf.nn.softmax(logits)
# The predicted class can be calculated from the y_pred matrix by taking the index of the largest element in each row
y_pred_cls = tf.argmax(y_pred, dimension=1)