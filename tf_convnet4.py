# Sentdex's version
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool2d(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutional_neural_network(data):
	# 5x5 convolutional filter, 1 input, 32 features
	weights = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
			   'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
			   'w_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
			   'out': tf.Variable(tf.random_normal([1024, n_classes]))}

	biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
			   'b_conv2': tf.Variable(tf.random_normal([64])),
			   'b_fc': tf.Variable(tf.random_normal([1024])),
			   'out': tf.Variable(tf.random_normal([n_classes]))}

	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	conv1 = tf.nn.relu(conv2d(x, weights['w_conv1']) + biases['b_conv1'])
	conv1 = max_pool2d(conv1)

	conv2 = tf.nn.relu(conv2d(conv1, weights['w_conv2']) + biases['b_conv2'])
	conv2 = max_pool2d(conv2)

	fc = tf.reshape(conv2, [-1, 7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['w_fc']) + biases['b_fc'])

	output = tf.matmul(fc, weights['out']) + biases['out']
	return output