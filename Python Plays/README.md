## CNNs Try Driving
---
This folder contains files that try to drive a car on [this track](https://github.com/ad71/Unity-Projects-2/tree/master/Car%20AI%203)
<br>
Here's what the files do (in no particular order).
<br>
1.  [screen_grab_low_res](https://github.com/ad71/Practical-ML/blob/master/Python%20Plays/screen_grab_low_res.py): 
Grabs a continuous stream of images (and the key held on that frame) in the region (0, 40, 800, 640) using OpenCV. 
The image is then converted to a resolution of 36 x 27 and saved into a numpy array.
2.  [screen_grab](https://github.com/ad71/Practical-ML/blob/master/Python%20Plays/screen_grab.py):
Grabs a continuous stream of images (and the key held on that frame) in the region (0, 40, 800, 640) using OpenCV.
The image is then converted to a resolution of 80 x 60 and saved into a numpy array.
3.  [screen_grab_faster](https://github.com/ad71/Practical-ML/blob/master/Python%20Plays/screen_grab_faster.py):
Grabs a continuous stream of images from a specified region using the `win32api`, `win32ui`, `win32con` modules. Found on StackOverflow.
4.  [balance_data](https://github.com/ad71/Practical-ML/blob/master/Python%20Plays/balance_data.py):
Equalizes the number of data points for each label. Optionally also shows the shuffled dataset.
5.  [get_keys](https://github.com/ad71/Practical-ML/blob/master/Python%20Plays/get_keys.py):
The `key_check` function in this module returns the current pressed key.
6.  [convnets](https://github.com/ad71/Practical-ML/blob/master/Python%20Plays/convnets.py):
Defines a few simple CNN models to try. All are modifications or simplifications of AlexNet
7.  [train_model](https://github.com/ad71/Practical-ML/blob/master/Python%20Plays/train_model.py):
Trains a model and saves a checkpoint.
8.  [test_model](https://github.com/ad71/Practical-ML/blob/master/Python%20Plays/test_model.py):
Loads a trained model from a checkpoint and tests against a currently running instance of the game.
9.  [tf_convnet_utils](https://github.com/ad71/Practical-ML/blob/master/Python%20Plays/tf_convnet_utils.py):
Helper functions for TensorFlow CNN models.
