function W = debugInitializeWeights(fan_out, fan_in)
	W = zeros(fan_out, 1 + fan_in);

	% Initialize weights using the sine function. This ensures that W always has the same values and will be useful for debugging
	W = reshape(sin(1:numel(W)), size(W)) / 10;
end;