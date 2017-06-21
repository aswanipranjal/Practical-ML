function J = computeCost(X, y, theta)
	m = length(y);
	J = 0;
	H = X*theta;
	J = 1/(2*m)*(H - y)'*(H - y);
end;