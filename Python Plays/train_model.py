# train_model.py

import numpy as np
from convnets import convnet1

width = 80
height = 60
lr = 1e-3
epochs = 8
model_name = 'pythondrives-{}-{}-{}-epochs.model'.format(lr, 'alexnetv0.01', epochs)

model = convnet1(width, height, lr)

train_data = np.load('C:\\Users\\Aman Deep Singh\\Documents\\Python\\Car CNN\\final_training_data.npy')
train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, width, height, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, width, height, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=epochs, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=model_name)

# tensorboard --logdir=foo:C:/Users/Aman Deep Singh/Anaconda3/log

model.save(model_name)