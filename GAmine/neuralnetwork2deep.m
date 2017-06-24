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
load('data_150.mat');
X = X(:, 2:end);
% size(X)
load('converted_fitness_150.mat');
m = size(X, 1);
fprintf('Datasets loaded\n');
fprintf('Press any key to continue\n');
pause;

% Loading parameters
load('preTheta1_50x19.mat');
load('preTheta2_50x51.mat');
load('preTheta3_12x51.mat');
% nn_params now has three matrices
nn_params = [Theta1(:); Theta2(:); Theta3(:)];
fprintf('Loaded saved neural network parameters\n');
fprintf('Press any key to continue\n');
pause;

fprintf('Feed forward\n');
lambda = 0;

J = nn2lCostFunction(nn_params, input_layer_size, hidden_1_layer_size, hidden_2_layer_size, num_labels, X, y, lambda);
fprintf('Cost at parameters: %f\n', J);
fprintf('Press any key to continue.\n');
pause;

fprintf('Initializing neural network parameters\n');
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_1_layer_size);
initial_Theta2 = randInitializeWeights(hidden_1_layer_size, hidden_2_layer_size);
initial_Theta3 = randInitializeWeights(hidden_2_layer_size, num_labels);

% Unroll parameters
initial_nn_parameters = [initial_Theta1(:); initial_Theta2(:); initial_Theta3(:)];

% Checking gradients from backpropagation by numerical methods
% fprintf('Checking backpropagation\n');
% IDK if computeNumericalGradient will be similar for multiple layers

