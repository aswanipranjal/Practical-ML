# https://github.com/miga101/tf_mnist_cnn/blob/master/cnn_mnist_.py
import tf_convnet_utils as util
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
from datetime import timedelta
import os

img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

filter_size1 = 5
num_filters1=  16
filter_size2 = 5
num_filters2 = 36
fc_size = 128
batch_size = 256
train_batch_size = 64
best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 1000
total_iterations = 0

data = input_data.read_data_sets("tmp/data", one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

def define_cnn(x_image=None, num_channels=None, filter_size1=None, num_filters1=None, filter_size2=None, num_filters2=None, fc_size=None, num_classes=None):
	layer_conv1, weight_conv1 = util.new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
	print(layer_conv1.shape)

	layer_conv2, weight_conv2 = util.new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
	print(layer_conv2.shape)

	layer_flat, num_features = util.flatten_layer(layer_conv2)
	print(layer_flat.shape)
	print(num_features) # 1764

	layer_fc1 = util.new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)
	print(layer_fc1.shape)

	layer_fc2 = util.new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)
	print(layer_fc2.shape)

	y_pred = tf.nn.softmax(layer_fc2)
	y_pred_cls = tf.argmax(y_pred, dimension=1)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true))
	return y_pred, y_pred_cls, loss, weight_conv1, weight_conv2

y_pred, y_pred_cls, loss, weight_conv1, weight_conv2 = define_cnn(x_image=x_image, num_channels=num_channels, filter_size1=filter_size1, num_filters1=num_filters1, filter_size2=filter_size2, num_filters2=num_filters2, fc_size=fc_size, num_classes=num_classes)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
save_dir = 'C:\\Users\\Aman Deep Singh\\Documents\\Python\\Practical ML\\'
if not os.path.exists(save_dir):
	print('Save path does not exist')

save_path = os.path.join(save_dir, 'best_validation')
session = tf.Session()
util.init_variables(session)

def optimize(num_iterations):
	global total_iterations
	global best_validation_accuracy
	global last_improvement
	start_time = time.time()
	for i in range(num_iterations):
		total_iterations += 1
		x_batch, y_true_batch = data.train.next_batch(train_batch_size)
		feed_dict_train = {x: x_batch, y_true: y_true_batch}
		session.run(optimizer, feed_dict=feed_dict_train)
		if (i % 50 == 0) or (i == (num_iterations - 1)):
			acc_train = session.run(accuracy, feed_dict=feed_dict_train)
			acc_validation, _ = validation_accuracy()
			# if improvement
			