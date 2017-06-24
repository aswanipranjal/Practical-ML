function p = predict2l(Theta1, Theta2, Theta3, X)
	m = size(X, 1);
	X=  [ones(m, 1) X];
	pa1 = X;
	pz2 = pa1*Theta1';
	pa2 = [ones(m, 1) sigmoid(pz2)];
	pz3 = pa2*Theta2';
	pa3 = [ones(m, 1) sigmoid(pz3)];
	pz4 = pa3*Theta3';
	pa4 = sigmoid(pz4);
	[dummy, p] = max(pa4, [], 2);
	p = p.*10000;
end;