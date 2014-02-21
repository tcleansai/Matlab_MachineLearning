function [W] = PCA1(X)
%This is PCA function to calculate the projection matrix
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

%sample size
[nsmp,~]= size(X);

%mean of samples
mean_X = mean(X, 1);

% substract mean
X = X - repmat(mean_X,size(X,1),1);

matrix = X*X';

% calculate the eignvector and eignvalue
[U, V] = eigs(matrix,nsmp-1);

% calculate projection matrix
W = X'*U*diag(diag(V).^(-0.5));

end

