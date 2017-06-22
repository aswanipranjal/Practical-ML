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
num_iters = 10000;
theta = zeros(19, 1);
% size(theta)
% The computCost function and the computeGradient functions written in the previous linear regression example were well vectorized, so we can directly use them.
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result:
fprintf('Theta computed from gradient descent: \n');
fprintf('%f\n', theta);
fprintf('\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Predicting stuff\n');
x_t = [1,0.77824,0.05768192,0.8901123,0.6041341,0.1500494,0.5331748,0.9752828,0.7247745,0.02214646,0.3577529,0.0419383,0.7006501,0.5528352,0.8018743,0.8233464,0.8771472,0.6299444,0.09685636];
y_t = [35325];
h_t = x_t*theta;
fprintf('Predicted track time: %f\n', h_t);
fprintf('Expected value: %f\n', y_t);

fprintf('Predicting stuff of test data set\n');
x_test = load('data.txt');
y_test = load('fitness.txt');
% Dropping the first column
x_test = x_test(:, 2:end);
y_test = y_test(:, 2:end);
x_test = [ones(size(x_test, 1), 1) x_test];
h_test = x_test*theta;
fprintf('Predicted		Expected\n');
[h_test y_test]