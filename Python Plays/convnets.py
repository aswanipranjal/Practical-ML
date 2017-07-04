import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Alexnet with reduced sizes of fully connected layers
# Network architecture: input -> conv -> max_pool -> conv -> max_pool -> conv -> conv -> conv -> max_pool -> fc -> fc -> fc(softmax) -> output
def convnet1(width, height, lr, output=4):
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
	model = tflearn.DNN(network, checkpoint_path='C:\\Users\\Aman Deep Singh\\Documents\\Python\\Practical ML\\Python Plays\\Checkpoints\\convnets_convnet1', max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
	return model

# A simplified network to test if things are working
def convnet2(width, height, lr, output=4):
	network = input_data(shape=[None, width, height, 1], name='input')
	network = conv_2d(network, 64, 5, strides=2, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	# No local response normalization
	network = conv_2d(network, 128, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = fully_connected(network, 256, activation='tanh')
	network = fully_connected(network, output, activation='softmax')
	network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=lr, name='targets')
	model = tflearn.DNN(network, checkpoint_path='C:\\Users\\Aman Deep Singh\\Documents\\Python\\Practical ML\\Python Plays\\Checkpoints\\convnets_convnet2', max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
	return model