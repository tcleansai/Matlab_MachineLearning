function [ W ] = LDA1(X,Y)
%This is LDA function to calculate the projection matrix
%
% It is only suitable for the situation that the sample number is
% less or equal dimension number
%
%Input:
%      X is the input data source
%           where the each row is a sample and each column is a feature
%      Y is the the label of X
%Output:
%      W is the projection matrix
%           size of W should be nfea*(nsmp-1) where nfea is the number of
%           feature
%
% F: the number of data dimension
% N: number of data sample
% C: number of labels
%
%Writtern by Wenyang Cai, Feb 21, 2013
%

% sample size
[nsmp, ~] = size(X);

% X substract its mean
X = X - repmat(mean(X),nsmp,1);

% labels
class = unique(Y);

% number of labels
nclass = length(class);

% initial and calculate number of samples for each label
nsmp_class = zeros(size(class));
for i = 1:length(class)
    nsmp_class(i) = sum(Y==class(i));
end

%{
formulat the problem:
W = maximize tr[W'X'MXW] 
s.t. W'X(I-M)XW = I

As M is a idempotent matrix, I - M is also a idempotent
===>
W = maximize tr[W'X'M'MXW] 
s.t. W'X(I-M)'(I-M)XW = I
Assume W = Q*U

===>
W = maximize tr[Q'U'X'M'MXUQ] 
s.t. Q'Q = I and Q'X(I-M)'(I-M)XU = I

first solve Q'X(I-M)'(I-M)XU = I to get U
then apply eigenanalysis to Q'U'X'M'MXUQ
choose the positive value
%}

% form M matrix
M = [];
for i = 1:length(class)
    M = blkdiag(M,ones(nsmp_class(i))/nsmp_class(i));
end

% X_w is F*N matrix
X_w = X'*(eye(nsmp)-M);

% Only choose the largest 
% U_w is N*(N-C) matrix V_w is (N-C)*(N-C)
[U_w,V_w] = eigs(X_w'*X_w,nsmp-nclass);

U = X_w*U_w*inv(V_w);% U is F*(N-C)

%X_b is (N-C)*N
X_b = U'*X'*M;

% Q is (N-C)*(C-1)
[Q,~] = eigs(X_b*X_b',nclass-1);

% final W
W = U*Q;
end

