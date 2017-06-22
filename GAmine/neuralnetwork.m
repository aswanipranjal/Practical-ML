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

% Pre-initialized weight matrices should be of these dimensions:
% Theta1 = [25x19]
% Theta2 = [12x26]

% Initialization
clear; close all; clc
input_layer_size = 18; % 18 genetically alterable parameters
hidden_layer_size = 25; % 25 hidden units
num_labels = 12; % 12 labels, from 0 - 120000 in multiples of 10000

% Loading and visualizing the datasets
fprintf('Loading datasets\n');
load('data_80.mat');
X = X(:, 2:end);
% size(X)
% loads the one-hot vectorized labelled dataset
load('converted_fitness_80.mat');
% size(y)
m = size(X, 1);
fprintf('Loaded datasets\n');
fprintf('Program has been paused. Press enter to continue\n');
pause;

% Loading parameters
load('preTheta1.mat');
load('preTheta2.mat');
nn_params = [Theta1(:); Theta2(:)];
% Matrices are 'ravel'ed into one vector
fprintf('Loaded saved neural network parameters\n');
fprintf('Program has been paused. Press enter to continue\n');
pause;

fprintf('Feed forward using neural network\n');
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at parameters: %f\n', J);
fprintf('Program has been paused. Press enter to continue\n');
pause;

% Regularization
fprintf('Checking function with regularization\n');
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at parameters: %f\n', J);
fprintf('Program has been paused. Press enter to continue\n');
pause;

% Checking if sigmoidGradient is valid
% fprintf('Evaluating sigmoidGradient\n');
% g = sigmoidGradient([-1, -0.5, 0, 0.5, 1]);
% fprintf('Sigmoid gradient evaluated at [-1, -0.5, 0, 0.5, 1]: \n');
% fprintf('%f', g);
% fprintf('\n\n');
% fprintf('Program has been paused. Press enter to continue\n');
% pause;

fprintf('Initializing neural network parameters\n');
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];

% Checking gradients from backpropagation by numerical methods
% fprintf('\nChecking backpropagation\n');
% checkNNGradients;
% fprintf('Program has been paused. Press enter to continue\n');
% pause;

% Checking gradients from backpropagation (with regularization)
% fprintf('\nChecking backpropagation (with regularization)\n');
% Check gradients by running checkNNGradients
% lambda = 3;
% checkNNGradients(lambda);

% Output cost function debug values
% debug_J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
% fprintf('Cost at (fixed) debugging parameters (w/ lambda = %f): %f\n\n', lambda, debug_J);

% fprintf('Program has been paused. Press enter to continue.\n');
% pause;

% Training Neural Network
fprintf('Training neural network\n');
options = optimset('MaxIter', 400);
lambda = 0;

costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

fprintf('Program has been paused. Press enter to continue.\n');
pause;

fprintf('Predicting a value: \n');
x_test = load('data.txt');
x_test = x_test(:, 2:end);
y_test = load('fitness.txt');
y_test = y_test(:, 2);
pred = predict(Theta1, Theta2, x_test);
fprintf('The predicted values are: \n');
afm = [pred y_test];
difference = abs(afm(:, 1) - afm(:, 2));
booldiff = difference > 10000;
test_cases = size(x_test, 1);
accuracy = (test_cases - sum(booldiff))/test_cases*100;
% disp(afm);
% disp(booldiff);
fprintf('Accuracy: %f', accuracy);