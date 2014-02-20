function [ W ] = LDA1(X,Y)
%LDA1 Summary of this function goes here
%   Detailed explanation goes here
[nsmp, ~] = size(X);
X = X - repmat(mean(X),nsmp,1);
class = unique(Y);
nclass = length(class);
nsmp_class = zeros(size(class));
for i = 1:length(class)
    nsmp_class(i) = sum(Y==class(i));
end

M = [];
for i = 1:length(class)
    M = blkdiag(M,ones(nsmp_class(i))/nsmp_class(i));
end

X_w = X'*(eye(nsmp)-M);% X_w is F*N matrix

[U_w,V_w] = eig(X_w'*X_w);% U_w is N*(N-C) matrix V_w is (N-C)*(N-C)

[~,idx] = sort(diag(V_w),'descend');
idx = idx(1:(nsmp-nclass));
V_w = diag(V_w);
V_w = diag(V_w(idx));
U_w = U_w(:,idx);

U = X_w*U_w*inv(V_w);% U is F*(N-C)

X_b = U'*X'*M;%X_b is (N-C)*N

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q = PCA1(X_b');% Q is
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W = U*Q;
W = real(W);
end

