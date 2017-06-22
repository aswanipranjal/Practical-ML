function p = predict(Theta1, Theta2, X)
	m = size(X, 1);
	X = [ones(m, 1) X];
	% size(X)
	pa1 = X;
	pz2 = pa1*Theta1';
	pa2 = [ones(m, 1) sigmoid(pz2)];
	pz3 = pa2*Theta2';
	pa3 = sigmoid(pz3);
	% size(pa3)
	p = pa3;
	[dummy, p] = max(p, [], 2);
	p = p.*10000;
end;