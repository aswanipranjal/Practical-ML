import tensorflow as tf
import cv2
import pong_neural_network
import numpy as np
import random
from collections import deque

# Defining hyperparameters
ACTIONS = 3
GAMMA = 0.99
# update gradient or training time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# how many frames to anneal epsilon
