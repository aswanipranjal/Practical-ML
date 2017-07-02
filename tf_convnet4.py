# Sentdex's version
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def convolutional_neural_network(data):
	# 5x5 convolutional filter, 1 input, 32 features
	weights = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
			   'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
			   'w_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
			   'out': tf.Variable(tf.random_normal([1024, n_classes]))}

	