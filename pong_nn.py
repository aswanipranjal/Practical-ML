import tensorflow as tf
import cv2
import pong_neural_network
import numpy as np
import random
from collections import deque

# Defining hyperparameters
ACTIONS = 3
GAMMA = 0.99
# update gradient or training time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# how many frames to anneal epsilon
EXPLORE = 500000
OBSERVE = 50000
REPLAY_MEMORY = 50000
# batch size
BATCH = 100

# Create tf graph (5 layer CNN in tensorflow)
def create_graph():
	# first convolutional layer
	w_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
	b_conv1 = tf.Variable(tf.zeros[32])

	# second convolutional layer
	w_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
	b_conv2 = tf.Variable(tf.zeros[64])

	# third convolutional layer
	w_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
	b_conv3 = tf.Variable(tf.zeros[64])

	# fourth fully connected layer
	w_fc4 = tf.Variable(tf.zeros([784, ACTIONS]))
	b_fc4 = tf.Variable(tf.zeros[784])

	# fifth fully connected layer
	w_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
	b_fc5 = tf.Variable(tf.zeros[[ACTIONS]])

	# input for pixel data
	s = tf.placeholder('float', [None, 84, 84, 84])

	# compute relu activation function on 2d convolutions given 4d inputs and filter tensors
	conv1 = tf.nn.relu(tf.nn.conv2d(s, w_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1)
	conv2 = tf.nn.relu(tf.nn.conv2d(s, w_conv2, strides=[1, 4, 4, 1], padding='VALID') + b_conv2)
	conv3 = tf.nn.relu(tf.nn.conv2d(s, w_conv3, strides=[1, 4, 4, 1], padding='VALID') + b_conv3)
	conv3_flat = tf.reshape(conv3, [-1, 3136])
	fc4 = tf.nn.relu(tf.matmul(conv3_flat, w_fc4 + b_fc4))
	fc5 = tf.matmul(fc5, w_fc5) + b_fc5
	return s, fc5

