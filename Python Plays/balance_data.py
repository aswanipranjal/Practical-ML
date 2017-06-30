import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('C:\\Users\\aman Deep Singh\\Documents\\Python\\Car CNN\\final_training_data.npy')
print(len(train_data))
# Loop to show training data
# for data in train_data:
# 	img = data[0]
# 	choice = data[1]
# 	cv2.imshow('test', img)
# 	print(choice)
# 	if cv2.waitKey(25) & 0xFF == ord('q'):
# 		cv2.destroyAllWindows()
# 		break

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))
# To balance the data, we will truncate the data to match the length of the lowest list (right in this case)

lefts = []
rights = []
forwards = []
backwards = []

# We shuffle the data so that the network is not biased for a particular output
shuffle(train_data)

for data in train_data:
	img = data[0]
	choice = data[1]

	if choice == [1, 0, 0, 0]:
		lefts.append([img, choice])
	elif choice == [0, 1, 0, 0]:
		forwards.append([img, choice])
	elif choice == [0, 0, 1, 0]:
		rights.append([img, choice])
	elif choice == [0, 0, 0, 1]:
		backwards.append([img, choice])
	else:
		print("No matches!")

# To make sure they are all of the same length
forwards = forwards[:len(lefts)][:len(rights)][:len(backwards)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]
backwards = backwards[:len(forwards)]

final_data = forwards + lefts + rights + backwards
shuffle(final_data)
print(len(final_data))
np.save('C:\\Users\\Aman Deep Singh\\Documents\\Python\\Practical ML\\Python Plays\\final_training_data_v2.npy', final_data)