function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
	m = length(y);
	J_history = zeros(num_iters, 1);

	for iter = 1:num_iters
		H = X * theta;
		errors = H - y;
		sigma_term = X' * errors; % [2x1]
		theta = theta - (alpha / m) * sigma_term;
		J_history(iter) = computeCost(X, y, theta);
	end;
end;