function plotData(X, y)
	figure; % opens a new figure window
	plot(X, y, 'kx', 'MarkerSize', 5);
	xlabel('Car number');
	ylabel('Track record time');
end;