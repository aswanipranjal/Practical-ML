function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
	m = length(y);
	J_history = zeros(num_iters, 1);

	for iter = 1:num_iters,
		H = X * theta;
		dH = H - y;
		S = X' * dH;
		theta = theta - (alpha/m) * S;
		J_history(iter) = computeCostMulti(X, y, theta);
	end;
end;