clear; close all; clc
fprintf('Loading data\n');
load('X.mat');
load('y.mat');
% Checking if data has loaded
% size(X)
% size(y)
fprintf('Data loaded\n');
% Discarding first columns, as they are the car numbers and not real data
X = X(:, 2:end);
y = y(:, 2:end);
m = length(y);

fprintf('First 10 examples from the dataset: \n');
disp(X(1:10, :));
disp(y(1:10, :));
fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('Features are normalized\n');
X = [ones(m, 1) X];
fprintf('Added a column of ones\n');
% disp(X(1:10, :));

fprintf('Running gradient descent\n');
% Setting hyperparameters
alpha = 0.01;
num_iters = 400;
theta = zeros(19, 1);
size(theta)