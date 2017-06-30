# A convolutional neural network for classifying images in the CIFAR-10 dataset. It also shows how to use different
# networks during training and testing
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt

print("TensorFlow version: {}".format(tf.__version__))
print("PrettyTensor version: {}".format(pt.__version__))

import cifar10
# A 163 MB download
cifar10.data_path = "tmp/data/CIFAR-10/"