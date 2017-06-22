function [vecY] = convertLabelsAnn(y)
	% Assuming y has 2 columns
	y = y(:, 2:end);
	size(y)
	m = length(y);
	vecY = zeros(m, 12);
	I = eye(12);
	for i = 1:m,
		if (y(i)>=0 && y(i) <= 10000), 
			vecY(i, :) = I(1, :);
		elseif (y(i, 1) > 10000 && y(i, 1) <= 20000),
			vecY(i, :) = I(2, :);
		elseif (y(i, 1) > 20000 && y(i, 1) <= 30000),
			vecY(i, :) = I(3, :);
		elseif (y(i, 1) > 30000 && y(i) <= 40000),
			vecY(i, :) = I(4, :);
		elseif (y(i, 1) > 40000 && y(i) <= 50000),
			vecY(i, :) = I(5, :);
		elseif (y(i, 1) > 50000 && y(i) <= 60000),
			vecY(i, :) = I(6, :);
		elseif (y(i, 1) > 60000 && y(i) <= 70000),
			vecY(i, :) = I(7, :);
		elseif (y(i, 1) > 70000 && y(i) <= 80000),
			vecY(i, :) = I(8, :);
		elseif (y(i, 1) > 80000 && y(i) <= 90000),
			vecY(i, :) = I(9, :);
		elseif (y(i, 1) > 90000 && y(i) <= 100000),
			vecY(i, :) = I(10, :);
		elseif (y(i, 1) > 100000 && y(i) <= 110000),
			vecY(i, :) = I(11, :);
		elseif (y(i, 1) > 110000),
			vecY(i, :) = I(12, :);
		end;
	end;
end;