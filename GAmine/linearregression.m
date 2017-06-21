% The script that controls the parameters for linear regression
clear; close all; clc

fprintf('Loading data matrices\n');
load('X.mat');
% size(X)
load('y.mat');
% size(y)
fprintf('Matrices loaded\n');
X = X(:, 2:end);
% plot y data here to see if things are loading
% plotData(y);
y = y(:, 2:end);
% fprintf('Split linear regression training data\n');
