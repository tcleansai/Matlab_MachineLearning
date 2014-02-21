function [ W ] = ICA1(X)
%ICA Summary of this function goes here
% Input:  
%       X is a N*F dimension matrix
%           column represents the features
%           row represents the observations
%
% Output:
%       W is the weight matrix
%


% Predefine constant
a1 = 1; %  1 <= a1<=2

% get sample size and feature number
[nsmp, nfea] = size(X);

% Centering
X = X - repmat(mean(X),nsmp,1);

% Whitening
U1 = WPCA1(X);
X_w = X*U1;


[nsmp, nfea] = size(X_w);
% Random a initial value of weight vector w
W = rand(nfea);

for i = 1:nfea
    converged = false;
    ii = 0;
    while ~converged
        ii = ii + 1;
        % take one w from w matrix
        w_init = W(i,:);
        
        %normalize w_init
        w_init = w_init/norm(w_init);
        
        % take one sample
        x = X_w;
        
        % calculate w = (xg(w'x))-(g'(w'x))w
        w_new = (mean(x'*(g1(w_init*x',a1))') - mean(g2(w_init*x',a1))*w_init')';
        
        % calcualte new w_new = w/|w|
        w_new = w_new/norm(w_new);
        
        W(i,:) = w_new;
        
        % perform Gram Schmidt method
%         W_GS = w_new;
%         if i > 1
%             tmp = 0;
%             for j = 1:i-1
%                 tmp = tmp + W(j,:)*W(j,:)'*W_GS; 
%             end
%             W_GS = W_GS - tmp;
%             W_GS = W_GS/((W_GS*W_GS').^0.5);
%         end
%         % normalize vector
%         W_GS = W_GS/norm(W_GS);
%         
%         %1 - abs(W_GS*w_new')
%         if 1 - abs(W_GS*w_new') < 0.01
            converged = true;
%         end
%         if i > 1
%             W(i,:) = W_GS;
%         end
    end
end
%de whitening
W = U1*W;
end

function [result_g1] = g1(X,a1)
    result_g1 = tanh(a1*X);
end

function [result_g2] = g2(X,a1)
    result_g2 = a1-a1*tanh(a1*X).^2;
end
