function [label] = mlrPredict(W, X)
% blrObjFunction predicts the label of data given the data and parameter W
% of multi-class Logistic Regression
%
% Input:
% W: the matrix of weight of size (D + 1) x 10
% X: the data matrix of size N x D
%
% Output: 
% label: vector of size N x 1 representing the predicted label of
%        corresponding feature vector given in data matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
label = zeros(size(X, 1), 1); % dummy return
input_count = size(X,1);
X = [ones(input_count,1) X];
a = X*W;
expa = exp(a);
se = sum(expa,2);
semat = repmat(se,1,10);
y = expa./semat;

[val, label] = max(y, [], 2);

end

