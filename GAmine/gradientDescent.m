function [theta, J_history] = gradientDescen(X, y, theta, alpha, num_iters)
	m = length(y);
	J_history = zeros(num_iters, 1);

	for iter = 1:num_iters,
		H = X * theta;
		sigma_term = X' * (X * theta); % [2x1]
		theta = theta - (alpha / m) * sigma_term;
		J_history = computeCost(X, y, theta);
	end;
end;