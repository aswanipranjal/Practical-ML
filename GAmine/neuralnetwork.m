% Setting up a three layer neural network to predict track record time
% First layer is the input layer with 19 inputs (including the one)
% Second layer is the hidden layer with 25 nodes, third layer is the output layer with outputs in this range
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
% Todo: convert all labels into one-hot vectors

% Initialization
clear; close all; clc
input_layer_size = 18; % 18 genetically alterable parameters
hidden_layer_size = 25; % 25 hidden units
num_labels = 12; % 12 labels, from 0 - 120000 in multiples of 10000

% Loading and visualizing the datasets
fprintf('Loading datasets\n');
load('data_80.mat');
% size(X)
load('converted_fitness_80.mat');
% size(y)
m = size(X, 1);
fprintf('Loaded datasets\n');
fprintf('Program has been paused. Press enter to continue\n');
pause;
