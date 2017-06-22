function checkNNGradients(lambda)
	if ~exist('lambda', 'var') || isempty(lambda)
		lambda = 0;
	end;

	% Define our own parameters and check if backpropagation is giving us the correct gradients
	input_layer_size = 3;
	hidden_layer_size = 5;
	num_labels = 3;
	m = 5;

	% We generate some 'random' test-data
	Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
	Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);

	X = debugInitializeWeights(m, input_layer_size - 1);
	y = 1 + mod(1:m, num_labels)';

	% Unroll parameters
	nn_params = [Theta1(:) ; Theta2(:)];

	% Short-hand for cost function
	costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

	[cost, grad] = costFunc(nn_params);
	numgrad = computeNumericalGradient(costFunc, nn_params);

	% Visually examine the two gradient computations. the two columns we get should be very similar
	disp([numgrad grad]);
	fprintf(['The above two columns you get should be very similar.\n' '(Left-Your Numerical Gradient, Right Analytical Gradient)\n\n']);

	% evaluate the norm of the difference between the two solutions.
	% If you have a correct implementation, and assuming you used EPSILON = 0.0001
	% in computeNumericalGradient.m, then the diff below should be less than 1e-9
	diff = norm(numgrad - grad)/norm(numgrad + grad);

	fprintf(['If backpropagation implementation is correct, then \n' ...
			 'the relative difference will be small (less than 1e-9). \n' ...
			 '\nRelative Difference: %g\n'], diff);
end;