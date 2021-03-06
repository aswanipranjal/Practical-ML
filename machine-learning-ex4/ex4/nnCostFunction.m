function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Does not work
% Y = zeros(m, num_labels);
% a1 = [ones(m, 1) X];
% z2 = a1*Theta1';
% a2 = [ones(m, 1) sigmoid(z2)];
% z3 = a2*Theta2';
% h = sigmoid(z3);
% H = sum(h, 2);
% for i = 1:m,
% 	for j = 1:num_labels,
% 		Y(i, j) = (y(i) == j);
% 	end;
% end;
% Y = sum(Y, 2);
% J = (1/m)*(-log(h)'*Y - log(1 - H)'*(1 - Y));

% X = [ones(m, 1) X];
% for i = 1:m
% 	x_i = X(i,:);
% 	h_i = sigmoid([1 sigmoid(x_i * Theta1')] * Theta2');
% 	y_i = zeros(1, num_labels);
% 	y_i(y(i)) = 1;

	% J = J + sum(-1 * y_i .* log(h_i) - (1 - y_i) .* log(1 - h_i));
% end;
% J = 1 / m * J;

% Regularization
% t1 = Theta1(:, 2:end);
% t2 = Theta2(:, 2:end);
% J = J + lambda/(2*m) * (sum(sumsq(t1)) + (sum(sumsq(t2))));

% Backpropagation
% Very confusing for loop approach
% DELTA1 = zeros(size(Theta1));
% DELTA2 = zeros(size(Theta2));
% for t = 1 : m,
% 	a1 = X(t,:);
% 	z2 = a1*Theta1';
% 	a2 = [1 sigmoid(z2)];
% 	z3 = a2*Theta2';
% 	a3 = sigmoid(z3);
% 	delta3 = zeros(num_labels);
% 	y_i = zeros(1, num_labels);
% 	yi(y(t)) = 1;
% 	delta3 = a3 - y_i;
% 	delta2 = delta3*Theta2 .* sigmoidGradient([1 z2]);
% 	DELTA1 = DELTA1 + delta2(2:end)' * a1;
% 	DELTA2 = DELTA2 + delta3' * a2;
% end;

% Theta1_grad = DELTA1 / m;
% Theta2_grad = DELTA2 / m;

% Regularization with cost function and gradients
% Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m * Theta1(:, 2:end);
% Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m * Theta2(:, 2:end);

% =================================================================================
% Remade, vectorizing wherever possible
X = [ones(m, 1) X];
yd = eye(num_labels);
y = yd(y, :);

a1 = X;
z2 = X*Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);

% Trying mobhai's method
% p = (log(a3)*y')+(log(1-a3)*(1-y)');
% p1 = sum(diag(p));
% p1 = -p1/m;
% J = p1;
singular_cost = (-y).*log(a3) - (1 - y).*log(1 - a3); % because y is now a matrix, use dot product

% Regularization with cost function and gradients
t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);
J = 1/m*sum(sum(singular_cost)) + lambda/(2*m)*(sum(sumsq(t1)) + sum(sumsq(t2)));
% J = J + lambda/(2*m)*(sum(sumsq(t1)) + sum(sumsq(t2)));

% Backpropagation vectorized
DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));
delta3 = a3 - y;
z2 = [ones(m, 1) z2];
delta2 = delta3*Theta2.*sigmoidGradient(z2);
delta2 = delta2(:, 2:end);
DELTA1 = DELTA1 + delta2'*a1; % same size as Theta1_grad (25 x 401)
DELTA2 = DELTA2 + delta3'*a2; % same size as Theta2_grad (10 x 26)
Theta1_grad = (1/m)*DELTA1;
Theta2_grad = (1/m)*DELTA2;

% Regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end