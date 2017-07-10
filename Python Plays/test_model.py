import numpy as np
import os
from screen_grab_faster import grab_screen

def main():
	for i in list(range(4))[::-1]:
		print(i+1)
		time.sleep(1)
	last_time = time.time()

	while True:
		screen = grab_screen(region=(0, 40, 800, 640))
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
		screen = cv2.resize(screen, (80, 60))
		keys = key_check()
		output = keys_to_output(keys)
		training_data.append([screen, output])
		# print('Frame took {} seconds'.format(time.time() - last_time))
		last_time = time.time()

main()