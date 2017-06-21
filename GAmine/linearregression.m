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

predict1 = [1, 0.8]*theta;
fprintf('For a max steering angle of 0.8(normalized), we predict a track record time of %f\n', predict1);
predict2 = [1, 0.5]*theta;
fprintf('For a max steering angle of 0.5(normalized), we predict a track record time of %f\n', predict2);
fprintf('Program paused. Press enter to continue.\n');

fprintf('Visualizing J(theta_0, theta_1)\n');

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
% I couldn't guess what the third argument does.
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% fill out J_vals
for i = 1:length(theta0_vals)
	for j = 1:length(theta1_vals)
		t = [theta0_vals(i); theta1_vals(j)];
		J_vals(i, j) = computeCost(X, y, t);
	end;
end;

J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals);
xlabel('\theta0'); ylabel('\theta1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-10, 10, 20))
xlabel('\theta0'); ylabel('\theta1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);