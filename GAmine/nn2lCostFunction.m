function [J grad] = nn2lCostFunction(nn_params, input_layer_size, hidden_1_layer_size, hidden_2_layer_size, num_labels, X, y, lambda)
	% This function assumes that the vectors are one-hot vectorized
	% Reshaping neural-network params vector into matrices
	Theta1 = reshape(nn_params(1:hidden_1_layer_size * (input_layer_size + 1)), hidden_1_layer_size, (input_layer_size + 1));
	Theta2 = reshape(nn_params((1 + (hidden_1_layer_size * (input_layer_size + 1))):(hidden_1_layer_size * (input_layer_size + 1))+(hidden_2_layer_size * (hidden_1_layer_size + 1))), hidden_2_layer_size, (hidden_1_layer_size + 1));
	Theta3 = reshape(nn_params((1 + (hidden_1_layer_size * (input_layer_size + 1))+(hidden_2_layer_size * (hidden_1_layer_size + 1))):end), num_labels, (hidden_2_layer_size + 1));
	m = size(X, 1);
	J = 0;

	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));
	Theta3_grad = zeros(size(Theta3));

	X = [ones(m, 1) X];
	% Layer 1
	z2 = X*Theta1';
	% Layer 2
	a2 = [ones(m, 1) sigmoid(z2)];
	z3 = a2*Theta2';
	% Layer 3
	a3 = [ones(m, 1) sigmoid(z3)];
	z4 = a3*Theta3';
	% Output layer
	a4 = sigmoid(z4);

	singular_cost = (-y).*log(a4) - (1 - y).*log(1 - a4);

	% Regularization with cost function and gradients
	t1 = Theta1(:, 2:end);
	t2 = Theta2(:, 2:end);
	t3 = Theta3(:, 2:end);
	J = 1/m*sum(sum(singular_cost)) + lambda/(2*m)*(sum(sumsq(t1)) + sum(sumsq(t2)) + sum(sumsq(t3)));
end;