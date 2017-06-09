from numpy import *

def compute_error_for_given_points(b, m, points):
	totalError = 0
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m * x + b))**2
	# we divide the sum by the total number of elements in the array
	return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
	# Gradient Descent

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate)

	return [b, m]

def run():
	# read dataset
	# points has [x, y] value pairs
	points = genfromtext('student_data.csv', delimiter=',')
	# hyperparameters
	learning_rate = 0.0001
	# y = mx + b
	initial_b = 0
	initial_m = 0
	# Number of iterations for training
	# we have a pretty small dataset, so we will iterate for 1000 cycles
	num_iterations = 1000
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print(b)
	print(m)

if __name__ = '__main__':
	run()