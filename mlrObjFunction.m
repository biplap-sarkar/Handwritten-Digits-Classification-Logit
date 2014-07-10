function [error, error_grad] = mlrObjFunction(W, X, T)
% mlrObjFunction computes multi-class Logistic Regression error function 
% and its gradient.
%
% Input:
% W: the vector of size ((D + 1) * 10) x 1. Later on, it will reshape into
%    matrix of size D + 1) x 10
% X: the data matrix of size N x D
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size ((D+1) * 10) x 1 representing the gradient 
%             of error function


W = reshape(W, size(X, 2) + 1, size(T, 2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error = 0; % dummy return
error_grad = zeros((size(X, 2) + 1) * 10, 1); % dummy return

input_count = size(X,1);
X = [ones(input_count,1) X];
a = X*W;
lse = logsumexp(a,2);
expa = exp(a);
se = sum(expa,2);
semat = repmat(se,1,size(T,2));
y = expa./semat;
lsemat = repmat(lse,1,size(T,2));
error = (-1)*sum(sum(T.*(log(expa)-lsemat)),2);

%error_grad = reshape(error_grad, size(X,2), size(T,2));
error_grad_k = X'*(y-T);
error_grad = error_grad_k(:);
%error_grad = zeros((size(X, 2) + 1) * 10, 1); % dummy return


end
