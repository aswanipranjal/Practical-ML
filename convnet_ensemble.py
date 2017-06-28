import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import time
import math
import os
import prettytensor as pt
from tensorflow.examples.tutorials.mnist import input_data

print('TensorFlow version: {}', tf.__version__)
print('PrettyTensor version: {}', pt.__version__)
data = input_data.read_data_sets("tmp/data", one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)
print(combined_images.shape)
print(combined_labels.shape)
combined_size = len(combined_images)
print(combined_size)
train_size = int(0.8 * combined_size)
validation_size = combined_size - train_size

# Helper function for splitting the data into random training and validation sets
def random_training_set():
	# Create a randomized index into the full/combined dataset
	idx = np.random.permutation(combined_size)
	# Split the random index into  training anf validation sets
	idx_train = idx[0:train_size]
	idx_validation = idx[train_size:]
	x_train = combined_images[idx_train, :]
	y_train = combined_labels[idx_train, :]
	x_validation = combined_images[idx_validation, :]
	y_validation = combined_labels[idx_validaiton, :]
	return x_train, y_train, x_validation, y_validation

