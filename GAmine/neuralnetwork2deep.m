% Setting up a four layer neural network to predict track record time
% First layer is the input layer with 18 inputs (excluding the bias input)
% Second is the first hidden layer with 50 nodes, Third layer is the second hidden layer with 50 nodes as well
% Fourth layer is the output layer with outputs in range: 
% Track record time | Output one-hot vector
% 0 - 10000 -> 		[1 0 0 0 0 0 0 0 0 0 0 0]
% 10001 - 20000 -> 	[0 1 0 0 0 0 0 0 0 0 0 0]
% 20001 - 30000 -> 	[0 0 1 0 0 0 0 0 0 0 0 0]
% 30001 - 40000 -> 	[0 0 0 1 0 0 0 0 0 0 0 0]
% 40001 - 50000 ->	[0 0 0 0 1 0 0 0 0 0 0 0]
% 50001 - 60000 -> 	[0 0 0 0 0 1 0 0 0 0 0 0]
% 60001 - 70000 -> 	[0 0 0 0 0 0 1 0 0 0 0 0]
% 70001 - 80000 -> 	[0 0 0 0 0 0 0 1 0 0 0 0]
% 80001 - 90000 -> 	[0 0 0 0 0 0 0 0 1 0 0 0]
% 90001 - 100000 -> [0 0 0 0 0 0 0 0 0 1 0 0]
% 100001 - 110000 ->[0 0 0 0 0 0 0 0 0 0 1 0]
% >110000 -> 		[0 0 0 0 0 0 0 0 0 0 0 1]

% Pre-initialized weight matrices should be of these dimensions
% Theta1 = [50x19]
% Theta2 = [50x51]
% Theta3 = [12x51]

% Initialization
clear; close all; clc
input_layer_size = 18;
hidden_1_layer_size = 50;
hidden_2_layer_size = 50;
num_labels = 12;

% Loading the datasets
fprintf('Loading datasets\n');
load('')