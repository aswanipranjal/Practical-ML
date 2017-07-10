import numpy as np
import cv2
import time
import os
from screen_grab_faster import grab_screen
from get_keys import key_check
import pyautogui
from convnets import convnet4

# use pyautogui.keyDown('_character_') and pyautogui.keyUp('_character_') to press keys

width = 80
height = 60
lr = 1e-3
epochs = 20
model_name = 'pythondrives-{}-{}-{}-epochs.model'.format(lr, 'thinalexv0.10', epochs)

def straight():
	pyautogui.keyDown('w')
	pyautogui.keyUp('a')
	pyautogui.keyUp('s')
	pyautogui.keyUp('d')
	# I don't know if we want to go straight even when we are turning

def left():
	pyautogui.keyUp('w')
	pyautogui.keyDown('a')
	pyautogui.keyUp('s')
	pyautogui.keyUp('d')

def right():
	pyautogui.keyUp('w')
	pyautogui.keyUp('a')
	pyautogui.keyUp('s')
	pyautogui.keyDown('d')

def brake():
	pyautogui.keyUp('w')
	pyautogui.keyUp('a')
	pyautogui.keyDown('s')
	pyautogui.keyUp('d')

model = convnet4(width, height, lr)
model.load(model_name)

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

		print('Frame took {} seconds'.format(time.time() - last_time))
		last_time = time.time()

		prediction = model.predict([screen.reshape(width, height, 1)])[0]
		moves = list(np.around(prediction))
		print(moves, prediction)
		
main()