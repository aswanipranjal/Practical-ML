import numpy as np

# function to calculate the sigmoid function. If deriv<>True, it returns the derivative
# The sigmoid fucntion has been chosen here because it has nice analytical features and is easy to teach with. In practice, large 
# scale deep-learning systems use piecewise-linear functions because they are much less expensive to calculate. ReLU for example
def nonlin(x, deriv=False):
	if(deriv==True):
		return (x*(1-x))

	return 1/(1 + np.exp(-x))

# input data
# The following code creates the input matrix. The first two inputs are the values for 'A' and 'B' respectively so that 
# we can predict A XOR B
# The third column is for accomodating the bias term and is not part of the input
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# output data
y = np.array([[0], [1], [1], [0]])

# We use this seed so that it gives the same random number every time we run it; which is sometimes useful for debugging
np.random.seed(1)

# synapses
# Now we initialize the weights to random values. syn0 are the weights between the input layer and the output layer. It is a 3x4 matrix
# because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4). syn1 are the weights between the 
# hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output. Note that there 
# is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends to 
# not work well when all the weights start at the same value.
syn0 = 2*np.random.random((3, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1

# training step
for j in range(600000):

	# calculate forward through the network
	l0 = X
	l1 = nonlin(np.dot(l0, syn0))
	l2 = nonlin(np.dot(l1, syn1))

	l2_error = y - l2

	if (j % 10000) == 0:
		print("Error: " + str(np.mean(np.abs(l2_error))))

	l2_delta = l2_error*nonlin(l2, deriv=True)
	# syn1.T probably returns the transpose matrix. not sure
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error*nonlin(l1, deriv=True)

	# update weights (irrespective of learning rate)
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

# If number of iterations in the training loop is increased, the final answer will be closer to the true output [0, 1, 1, 0]
print("Output after training")
print(l2)

# To do: extend this model to work with 3 inputs