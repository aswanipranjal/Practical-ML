import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('C:\\Users\\aman Deep Singh\\Documents\\Python\\Practical ML\\Python Plays\\training_data.npy')

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

# We shuffle the data so that the network is not biased for a particular output
shuffle(train_data)

for data in train_data:
	img = data[0]
	choice = data[1]

	if choice == [1, 0, 0]:
		lefts.append([img, choice])
	elif choice == [0, 1, 0]:
		forwards.append([img, choice])
	elif choice == [0, 0, 1]:
		rights.append([img, choice])
	else:
		print("No matches!")