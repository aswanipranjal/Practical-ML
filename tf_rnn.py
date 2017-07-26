# Humans also have temporal intelligence
# Traditional neural networks can not recognize sequences or orders
# RNNs are used for speech data or sequential data
# LSTM cells are used in RNNs
# LSTM cell has a keep/forget gate, sees wha it wants to add from the input and decides what to output
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hm_epochs = 10
n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def recurrent_neural_network(x):
	# Use truncated normal with standard deviation of 0.1 or 0.01 for better results
	layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}
	# Tensorflow wants us to input a transposed 3D matrix
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(0, n_chunks, x)
	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
	return output