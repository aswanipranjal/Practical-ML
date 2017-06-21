function plotData(y)
	figure; % opens a new figure window
	plot(y(:, 1), y(:, 2), 'k-', 'MarkerSize', 5);
	xlabel('Car number');
	ylabel('Track record time');
end;