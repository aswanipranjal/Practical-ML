from numpy import *

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations)

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