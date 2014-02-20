function [ W ] = LPP1(X)
%NPP1 Summary of this function goes here
%   Detailed explanation goes here
[nsmp, ~] = size(X);
X = X - repmat(mean(X),nsmp,1);
t = 10000000;
S = exp(-pdist2(X,X)/t);
D = diag(sum(S,2));
% W = QU
% argmin tr[W'X'(D-S)XW]
% s.t. W'X'DXW = I
% =>
% argmin tr[Q'U'X'(D-S)XUQ] 
% s.t. Q'Q = I
% U'X'DXU = I
%
% Because D is diagnal so d = D.^(0.5) and D = d*d?
%
% solve U'XddX'U = I

d = D.^(0.5);
X_w = X'*d;
[U_w, V_w] = eig(X_w'*X_w);
[~,idx] = sort(diag(V_w),'descend');
idx = idx(1:(nsmp-1));
V_w = diag(V_w);
V_w = diag(V_w(idx));
U_w = U_w(:,idx);
U = X_w*U_w*inv(V_w);
% solve argmin tr[Q'U'X'(D-S)X'UQ] to find Q

[Q,V_b] = PCA1(U'*X'*(D-S)*X*U);
Q = Q(:,(V_b>0));
W = U*Q;

end

