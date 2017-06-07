# 2 layer ANN
import numpy as np
import time

# variables
n_hidden = 10
n_input = 10
# outputs
n_output = 10
# sample_data
n_sample = 300

# hyperparameters
learning_rate = 0.01
momentum = 0.9

# we seed the random number generator (non-deterministic seeding)
np.randon.seed(0);

# activation function
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

# for XOR specifically, the tangent-prime performs much better
def tanh_prime(x):
	return 1 - np.tanh(x)**2

# x: input data, t: transpose (which will help us with matrix multiplications), V, W: 2 layers of our neural network, 
# bv, bw: 2 biases to apply on each network (biases help make a more accurate prediction)
def train(x, t, V, W, bv, bw):
	# froward propagation -- matrix multiply + biases
	A = np.dot(x, V) + bv
	# this is our first activation function
	Z = np.tanh(A)

	B = np.dot(Z, W) + bw
	# this is our second activation function
	Y = sigmoid(B)

	# Backward propagation
	Ew = Y - t
	Ev = tanh_prime(A) * np.dot(W, Ew)

	# predict our loss
	dW = np.outer(Z, Ew)
	dV = np.outer(x, Ev)

	# this is called cross-entropy. Generally for classification problems, cross-entropy tends to give us a better result
	loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))

	# this returns the loss, the delta values and the errors
	return loss, (dV, dW, Ev, Ew)

def prediction(x, V, W, bv, bw):
	A = np.dot(x, V) + bv
	B = np.dot(np.tanh(A), W) + bw
	return (sigmoid(B) > 0.5).astype(int)