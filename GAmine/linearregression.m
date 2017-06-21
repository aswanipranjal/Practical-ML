% The script that controls the parameters for linear regression
clear; close all; clc

fprintf('Loading data matrices\n');
load('X.mat');
% size(X)
load('y.mat');
% size(y)
fprintf('Matrices loaded\n');
X = X(:, 2:end); % separating the useful part of the data as the first column is just the car number
% plot y data here to see if things are loading
% plotData(y);
y = y(:, 2:end); % separating the useful part of the data as the first column is just the car number
% fprintf('Split linear regression training data\n');
% Implement linear regresion function for the first column of X only
m = length(y);
X = X(:, 1);
% plotData(X, y);
X = [ones(m, 1) X];
theta = zeros(2, 1); % [2x1]
iterations = 1500;
alpha = 0.01;
fprintf('\nTesting the cost function\n');
