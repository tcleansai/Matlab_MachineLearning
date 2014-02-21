function [ W ] = LPP1(X)
%This is LPP function to calculate the projection matrix
%
% It is only suitable for the situation that the sample number is
% less or equal dimension number
%
%Input:
%      X is the input data source
%           where the each row is a sample and each column is a feature
%Output:
%      W is the projection matrix
%           size of W should be nfea*(nsmp-1) where nfea is the number of
%           feature
%Writtern by Wenyang Cai, Feb 21, 2013
%

% sample size
[nsmp, ~] = size(X);

% substract 
X = X - repmat(mean(X),nsmp,1);

% choose to to make value in D not too large or small according to distance
t = 1000000.0;

% k = 10
% [idx] = knnsearch(X,X,'K',k+1);
% idx = idx(:,2:k+1);
% tmp = zeros(nsmp);
% for i = 1:nsmp
%     tmp(i,idx(i,:)) = 1;
% end
% tmp
% calculate the distance
dis = zeros(size(X,1));
for i = 1:size(X,1)
    for j = 1:size(X,1)
        tmp = X(i,:)-X(j,:);
        dis(i,j) = tmp*tmp';
    end
end
%dis = dis.^(0.5);

% calculate S
S = exp(-dis/t);

% calculate D matrix
D = diag(sum(S,2));

%{
formulat the problem:
W = maximize tr[W'X'(D-S)XW] 
s.t. W'XDXW = I
===>
As we have let d = D.^(0.5) and we have D = d*d'
W = maximize tr[W'X'(D-S)XW] 
s.t. W'Xd'dXW = I
===>
Assume W = Q*U
W = maximize tr[Q'U'X'M'MXUQ] 
s.t. Q'Q = I and Q'X'd'dXU = I

solve Q'X'd'dXU = I first to find U
and then find Q
%}

% calculate d
d = D.^(0.5);

% form X'd
X_w = X'*d;

% It should contain N-1 positive eigenvalues
[U_w, V_w] = eigs(X_w'*X_w,nsmp-1);

% calculate new U
U = X_w*U_w*diag(diag(V_w).^-1);

% solve argmin tr[Q'U'X'(D-S)X'UQ] to find Q
% as it's finding the minimum so we just select d eigenvector from small to
% large

%
matrix = U'*X'*(D-S)*X*U;

% get Q
[Q,~] = eig(matrix);

% Calculate W
W = U*Q;

end

