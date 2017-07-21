import numpy as np
import cv2
from screen_grab_faster import grab_screen
import time
import pyautogui
import os
from get_keys import key_check

def keys_to_output(keys):
	output = [0, 0, 0, 0]
	if 'A' in keys:
		output[0] = 1
	elif 'D' in keys:
		output[2] = 1
	elif 'W' in keys:
		output[1] = 1
	else:
		output[3] = 1
	return output

file_name = 'C:\\Users\\Aman Deep Singh\\Documents\\Python\\Car CNN\\training_data_36x27.npy'
if os.path.isfile(file_name):
	print('File exists, loading previous data')
	training_data = list(np.load(file_name))
else:
	print('File does not exist, starting fresh')
	training_data = []
