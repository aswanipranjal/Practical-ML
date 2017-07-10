import numpy as np
import cv2
import time
import os
from screen_grab_faster import grab_screen
from get_keys import key_check

def main():
	for i in list(range(4))[::-1]:
		print(i+1)
		time.sleep(1)
	last_time = time.time()

	while True:
		# get the input for the trained convolutional neural network
		screen = grab_screen(region=(0, 40, 800, 640))
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
		screen = cv2.resize(screen, (80, 60))
		# print('Frame took {} seconds'.format(time.time() - last_time))
		last_time = time.time()

main()