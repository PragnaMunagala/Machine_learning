function [final_cost, gradient] = cost_function(weight, X, y)
    m = length(y); 
    gradient = zeros(size(weight));
    h = 1 ./ (1 + exp(-1 * X * weight));
    final_cost = -(1 / m) * sum( (y .* log(h)) + ((1 - y) .* log(1 - h)) );
    for i = 1 : size(weight, 1)
        gradient(i) = (1 / m) * sum( (h - y) .* X(:, i) );
    end
end