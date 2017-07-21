import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Predicting future events based on popular past events per patient
train_dcast = pd.crosstab(index=[train['PID']], columns=train['Event'])
train_dcast.reset_index(drop=False, inplace=True)

train_dcast.head()