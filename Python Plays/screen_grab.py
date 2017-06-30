import numpy as np
# from PIL import ImageGrab
import cv2
from screen_grab_faster import grab_screen
import time
import pyautogui
import os
from get_keys import key_check

def keys_to_output(keys):
	# [A, W, D]
	output = [0, 0, 0]
	if 'A' in keys:
		output[0] = 1
	elif 'D' in keys:
		output[2] = 1
	else:
		output[1] = 1
	return output

# def draw_lines(image, lines):
# 	try:
# 		for line in lines:
# 			coords = line[0]
# 			cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)
# 	except:
# 		pass

# # define region of interest
# def roi(img, vertices):
# 	mask = np.zeros_like(img)
# 	cv2.fillPoly(mask, vertices, 255)
# 	masked = cv2.bitwise_and(img, mask)
# 	return masked

# def process_image(original_image):
# 	processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# 	processed_image = cv2.Canny(processed_image, threshold1=200, threshold2=300)
# 	processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)
# 	vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]])
# 	processed_image = roi(processed_image, [vertices])

# 	# Hough lines algorithm
# 	# If HoughLines is being used, makes sure to pass some sort of edge detection. In this case this will be Canny edge detection
# 	lines = cv2.HoughLinesP(processed_image, 1, np.pi/180, 180, np.array([]), 100, 15)
# 	draw_lines(processed_image, lines)
# 	return processed_image

# for i in list(range(4))[::-1]:
# 	print(i+1)
# 	time.sleep(1)

file_name = 'C:\\Users\\Aman Deep Singh\\Documents\\Python\\Practical ML\\Python Plays\\training_data.npy'
if os.path.isfile(file_name):
	print('File exists, loading previous data')
	training_data = list(np.load(file_name))
else:
	print('File does not exist, starting fresh')
	training_data = []

# last_time = time.time()
# while(True):
# 	screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
# 	new_screen = process_image(screen)
# 	# print('Down')
# 	# pyautogui.keyDown('w')
# 	# time.sleep(2)
# 	# print('Up')
# 	# pyautogui.keyUp('w')
# 	cv2.imshow('window', new_screen)
# 	# cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
# 	print('Loop took {} seconds'.format(time.time() - last_time))
# 	last_time = time.time()
# 	if cv2.waitKey(25) & 0xFF == ord('q'):
# 		cv2.destroyAllWindows()
# 		break

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
		print('Frame took {} seconds'.format(time.time() - last_time))
		last_time = time.time()
		if len(training_data) % 500 == 0:
			print(len(training_data))
			np.save(file_name, training_data)

main()
# Probably works