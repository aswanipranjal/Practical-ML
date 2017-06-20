from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		random.seed(1)
		self.synaptic_weights = 2 * random.random((3, 1)) - 1

	def sigmoid(self, x):
		return 1/(1 + exp(-x))

	def notsigmoid(self, x):
		return x * (1 - x)

	def predict(self, inputs):
		return self.sigmoid(dot(inputs, self.synaptic_weights))

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for i in range(number_of_training_iterations):
			output = self.predict(training_set_inputs)
			error = training_set_outputs - output
			adjustment = dot(training_set_inputs.T, error * self.notsigmoid(output))
			self.synaptic_weights += adjustment

if __name__ == '__main__':
	# Initialize a single neuron neural network
	neural_network = NeuralNetwork()
	print ('Random starting synaptic weights:')
	print (neural_network.synaptic_weights)

	# The training set. We have 4 examples, each consisting of 3 input values and 1 output value
	training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = array([[0, 1, 1, 0]]).T # The T function returns the transpose of the output set

	# train the neural netwrok using the training set.
	# Do it 10,000 times and make small adjustments each time
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print ('New synaptic weights after training: ')
	print (neural_network.synaptic_weights)

	# Test the neural network
	print ('Predicting')
	print (neural_network.predict(array([1, 0, 0])))