function [W,V] = LPP2(X)
%NPP1 Summary of this function goes here
%   Detailed explanation goes here
[nsmp, ~] = size(X);

X = X - repmat(mean(X),nsmp,1);

a=1


t = 10000;

S = exp(-pdist2(X,X)/t);

a=2

D = diag(sum(S,2));

form1 = X'*(D-S)*X;
form2 = X'*D*X;
matrix = form2\form1;

a = 3
matrix = (matrix+matrix')/2;


[U,V] = eig(matrix);

a = 4
W = U(:,diag(V)>0);
W = W(:,1:nsmp);

W= real(W);


size(W)
end

