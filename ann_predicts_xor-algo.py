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

# for XOR specifically, the inverse tangent performs much better