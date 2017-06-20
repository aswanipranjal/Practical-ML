function [J grad] = ann(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size*(input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Feed forward
X = [ones(m, 1) X];
y = eye(num_labels)(y, :);
a1 = X;
z2 = X * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Regularize
jcost = -y.*log(a3) - (1 - y).*log(1 - a3);
t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);
J = 1/m*sum(sum(jcost)) + lambda/(2*m)*(sum(sumsq(t1)) + sum(sumsq(t2)));

% Backpropagate
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
delta3 = a3 - y;
z2 = [ones(m, 1) z2];
delta2 = delta3 * Theta2 .* sigmoidGradient(z2);
delta2 = delta2(:, 2:end);
Delta1 = Delta1 + delta2' * a1;
Delta2 = Delta2 + delta3' * a2;
Theta1_grad = (1/m)*Delta1;
Theta2_grad = (1/m)*Delta2;

% Regularize
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);

% Unroll gradients
grad = [Theta1_grad(:); Theta2_grad(:)];
end;