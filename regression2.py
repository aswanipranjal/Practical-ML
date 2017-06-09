from numpy import *

def compute_error_for_given_points(b, m, points):
	totalError = 0
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m * x + b))**2
	# we divide the sum by the total number of elements in the array
	return totalError / float(len(points))

# Calculates gradient for one iteration
def step_gradient(b_current, m_current, points, learning_rate):
	# Gradient Descent
	b_gradient = 0
	m_gradient = 0
	# The number of points
	N = float(len(points))
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)
	return [new_b, new_m]

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