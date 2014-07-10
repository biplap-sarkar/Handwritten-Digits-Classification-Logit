function [w] = blrNewtonRaphsonLearn(initial_w, X, t, n_iter)
%blrNewtonRaphsonLearn learns the weight vector of 2-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_w: vector of size (D+1) x 1 where D is the number of features in
%            feature vector
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% t: vector of size N x 1 where each entry is either 0 or 1 representing
%    the true label of corresponding feature vector.
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% w: vector of size (D+1) x 1, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = zeros(size(X, 2) + 1, 1); % dummy return
w = initial_w;
input_count = size(X,1);
X = [ones(input_count,1) X];
for i=1:n_iter
    y = sigmoid(X*w);
    
    R = diag(sparse(y.*(1-y)));
    z = (X*w) - (inv(R) * (y-t));
    w = pinv(X'*R*X)*X'*R*z;
end
end
