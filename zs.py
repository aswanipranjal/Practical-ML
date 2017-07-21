import numpy as np
import pandas as pd

train = pd.read_csv("C:\\Users\\Aman Deep Singh\\Downloads\\2f97065a-6-YDS_dataset\\train.csv")
test = pd.read_csv("C:\\Users\\Aman Deep Singh\\Downloads\\2f97065a-6-YDS_dataset\\test.csv")

# Predicting future events based on popular past events per patient
train_dcast = pd.crosstab(index=[train['PID']], columns=train['Event'])
train_dcast.reset_index(drop=False, inplace=True)

print(train_dcast)