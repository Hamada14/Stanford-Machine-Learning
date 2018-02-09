function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
% You need to return the following variables correctly 
value_cal = (X * theta - y).^2;
J = sum(value_cal) / (2 * m);

end
