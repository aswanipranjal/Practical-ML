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

def main():
	for i in list(range(4))[::-1]:
		print(i+1)
		time.sleep(1)
	last_time = time.time()

	while True:
		screen = grab_screen(region=(0, 40, 800, 640))
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
		screen = cv2.resize(screen, (36, 27))
		keys = key_check()
		output = keys_to_output(keys)
		training_data.append([screen, output])
		last_time = time.time()
		if len(training_data) % 500 == 0:
			print(len(training_data))
			np.save(file_name, training_data)

main()