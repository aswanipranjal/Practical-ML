# train_model.py

import numpy as np
from convnets import convnet1, convnet2, convnet3, convnet4

width = 36
height = 27
lr = 1e-3
epochs = 20
# Removing dropouts
# model_name = 'pythondrives-{}-{}-{}-epochs.model'.format(lr, 'alexnetv0.02', epochs)
# model_name = 'pythondrives-{}-{}-{}-epochs.model'.format(lr, 'thinalexv0.01', epochs)
model_name = 'pythondrives-{}-{}-{}-epochs.model'.format(lr, '2deepCNN_low_res', epochs)

model = convnet2(width, height, lr)

train_data = np.load('C:\\Users\\Aman Deep Singh\\Documents\\Python\\Car CNN\\training_data_36x27.npy')
train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, width, height, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, width, height, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=epochs, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=model_name)

# tensorboard --logdir=foo:"C:/Users/Aman Deep Singh/Anaconda3/log"

model.save(model_name)