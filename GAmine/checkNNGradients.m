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
	Theta1 