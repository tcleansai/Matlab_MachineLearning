function [U, V] = PCA0(dataIn)
%This is a PCA used to calculate the
%Input:
%      dataIn is the input data source
%      dim is the original data would reduce
%Output:
%      U is the eignvector of covariance matrix
%      V is the eignvalue of the covariance matrix
%
%Writtern by Wenyang Cai, Feb 18, 2013
%

nsamp = size(dataIn,1);

mean_dataIn = mean(dataIn, 1);%mean of samples

% substract mean
data = dataIn - repmat(mean_dataIn,size(dataIn,1),1);

% calculate the covariance matrix
cov_matrix = 1/nsamp*(data'*data);
%cov_matrix = cov(dataIn)
% calculate the eignvector and eignvalue
[U, V] = eig(cov_matrix);
V = diag(V);

% sort the eignvalue and eignvector in descend order of eignvalue
[V, idx] = sort(V,'descend');
U = U(:,idx);
U = U(:,1:20);
end

