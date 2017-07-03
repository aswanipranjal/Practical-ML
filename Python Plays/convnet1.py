import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Alexnet
def convnet1(width, height, lr, output=3):
	network = input_data(shape=[None, width, height, 1], name='input')
	network = conv_2d(network, 96, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 256, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = fully_connected(network, 256, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 256, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, output, activation='softmax')
	network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=lr, name='targets')
	model = tflearn.DNN(network, checkpoint_path='C:\\Users\\Aman Deep Singh\\Documents\\Python\\Practical ML\\')