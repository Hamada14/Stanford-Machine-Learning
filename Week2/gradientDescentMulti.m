function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features

J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    temp_theta = theta;
    diff = X * theta - y;
    for feature = 1:n
      theta(feature) = temp_theta(feature) - alpha * diff'*X(:, feature)/m;
    end
    J_history(iter) = computeCostMulti(X, y, theta);
 end
end

