function [vecY] = convertLabelsAnn(y)
	% Assuming y has 2 columns
	y = y(:, 2:end);
	m = length(y);
	vecY = zeros(m, 12);
	I = eye(12);
	for i = 1:m,
		if (y(i) >= 0 and y(i) <= 10000), 
			vecY(i, :) = I(1, :);
		elseif (y(i, 1) > 10000 and y(i) <= 20000),
			vecY(i, :) = I(2, :);
		elseif (y(i, 1) > 20000 and y(i) <= 30000),
			vecY(i, :) = I(3, :);
		elseif (y(i, 1) > 30000 and y(i) <= 40000),
			vecY(i, :) = I(4, :);
		elseif (y(i, 1) > 40000 and y(i) <= 50000),
			vecY(i, :) = I(5, :);
		elseif (y(i, 1) > 50000 and y(i) <= 60000),
			vecY(i, :) = I(6, :);
		elseif (y(i, 1) > 60000 and y(i) <= 70000),
			vecY(i, :) = I(7, :);
		elseif (y(i, 1) > 70000 and y(i) <= 80000),
			vecY(i, :) = I(8, :);
		elseif (y(i, 1) > 80000 and y(i) <= 90000),
			vecY(i, :) = I(9, :);
		elseif (y(i, 1) > 90000 and y(i) <= 100000),
			vecY(i, :) = I(10, :);
		elseif (y(i, 1) > 100000 and y(i) <= 110000),
			vecY(i, :) = I(11, :);
		elseif (y(i, 1) > 110000),
			vecY(i, :) = I(12, :);
		end;
	end;
end;