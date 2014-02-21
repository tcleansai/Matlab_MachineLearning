function [ W ] = LPP1(X)
%NPP1 Summary of this function goes here
%   Detailed explanation goes here
[nsmp, ~] = size(X);

X = X - repmat(mean(X),nsmp,1);

t = 1000000.0;

dis = zeros(size(X,1));
for i = 1:size(X,1)
    for j = 1:size(X,1)
        tmp = X(i,:)-X(j,:);
        dis(i,j) = tmp*tmp';
    end
end
%dis = dis.^(0.5);

S = exp(-dis/t);

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

