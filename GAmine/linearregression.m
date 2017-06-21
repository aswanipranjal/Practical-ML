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
plotData(X, y);
X = [ones(m, 1) X];
theta = zeros(2, 1); % [2x1]
iterations = 1500;
alpha = 0.01;
fprintf('\nTesting the cost function\n');
J = computeCost(X, y, theta);
fprintf('With theta = [0, 0]\nCost computed = %f\n', J);
fprintf('Program paused. Press enter to continue\n');
pause;

fprintf('Running gradient descent\n');
theta = gradientDescent(X, y, theta, alpha, iterations);

% Printing theta
fprintf('Theta found by gradientDescent: \n');
fprintf('%f\n', theta);

% Plot the linear fit
hold on;
plot(X(:, 2), X*theta, '-')
legend('Training data', 'Linear Regression');
hold off;
