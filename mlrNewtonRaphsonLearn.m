function [W] = mlrNewtonRaphsonLearn(initial_W, X, T, n_iter)
%mlrNewtonRaphsonLearn learns the weight vector of multi-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_W: matrix of size (D+1) x 10 represents the initial weight matrix 
%            for iterative method
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% W: matrix of size (D+1) x 10, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%W = zeros(size(X, 2) + 1, 10); % dummy return

W = reshape(initial_W, size(X, 2) + 1, size(T, 2));
input_count = size(X,1);
X = [ones(input_count,1) X];
a = X*W;
expa = exp(a);
se = sum(expa,2);
semat = repmat(se,1,size(T,2));
y = expa./semat;

I = eye(10,10);
grad = X'*(y-T);
for k=1:10
    for j=1:10
        B = zeros(size(W,1),size(W,1));
%         Xn = X;
%         Xn = repmat(y(:,k).*(1-y(:,j)),1,size(X,2)).*Xn;
%         B = Xn'*X;
        for p=1:input_count
%            B = repmat(y(:,k).*(1-y(:,j)),1,size(X,2)) .* (X'*X);
            Xn = X(p,:);
            B = B + y(p,k)*(I(k,j)-y(p,j))*(Xn'*Xn);
%             if j==k
%                 B = B + y(p,k)*(1-y(p,j))*(Xn'*Xn);
%             else
%                 B = B + y(p,k)*(0-y(p,j))*(Xn'*Xn);
%             end
        end
        if j==1 && k==1
            H = B;
        else
            H = [H B];
        end
    end
end
grad = reshape(grad, 7160, 1);
%H = reshape(H,7160,7160);
% H = pinv(H);
% sub = H*grad;
% sub = reshape(sub,716,10);
% W = W - sub;

% Hl = reshape(H,1,7160*7160);
% C = reshape(Hl, 7160, 7160);
C = zeros(7160,7160);
for i = 1:10
    j = ((i-1)*716)+1;
    j1 = ((i)*716);
    k = ((i-1)*7160)+1;
    l = 7160*i;
    C(j:j1,:) = H(:,k:l);
end
C = pinv(C);
sub = C*grad;
sub = reshape(sub,716,10);
W = W - sub;


end