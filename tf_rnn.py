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

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	# Use truncated normal with standard deviation of 0.1 or 0.01 for better results
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_no]))}