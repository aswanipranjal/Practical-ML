function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
	% This function assumes that the values of y are one-hot vectorized
	% Reshaping weight vector into matrices
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
	m = size(X, 1);

	J = 0;
	% Initializing gradient matrices
	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));

	% Eliminating the first column.
	X = X(:, 2:end);
	X = [ones(m, 1) X];
	% Layer 1
	a1 = X;
	z2 = a1*Theta1';
	% Layer 2
	a2 = [ones(m, 1) sigmoid(z2)];
	z3 = a2*Theta2';
	% Output layer
	a3 = sigmoid(z3);

	singular_cost = (-y).*log(a3) - (1 - y).*log(1 - a3); % because y is now a matrix, we use the dot product

	% Regularization with cost function and gradients
	t1 = Theta1(:, 2:end);
	t2 = Theta2(:, 2:end);
	J = 1/m*sum(sum(singular_cost)) + lambda/(2*m)*(sum(sumsq(t1)) + sum(sumsq(t2)));

	% Backpropagation
	DELTA1 = zeros(size(Theta1));
	DELTA2 = zeros(size(Theta2));
	delta3 = a3 - y;
	z2 = [ones(m, 1) z2];
	delta2 = delta3*Theta2.*sigmoidGradient(z2);
	delta2 = delta2(:, 2:end);
	DELTA1 = DELTA1 + delta2'*a1; % same size as Theta1_grad [25x19]
	DELTA2 = DELTA2 + delta3'*a2; % same size as Theta2_grad [12x26]
	Theta1_grad = (1/m)*DELTA1;
	Theta2_grad = (1/m)*DELTA2;

	% Regularization
	Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
	Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);

	% Unroll gradients
	grad = [Theta1_grad(:); Theta2_grad(:)];
end;