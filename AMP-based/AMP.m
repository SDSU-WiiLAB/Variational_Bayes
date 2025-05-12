% AMP implementation for data detection
% H: channel matrix
% y: received signal
% N0: noise variance
% par: parameters for the implementation
% par.S: constellation
% par.ps(:,i): prior distribution for the constellation used by user-i, can be simply as 1/length(par.S)*ones(length(par.S), 1)
function [x_est, q, z, sigma_sq] = AMP(par, H, y, N0)
K = par.K;
N = par.const_size;
beta = K/par.M;
x_ = zeros(K,1);
for i=1:K
    x_(i) = sum(par.ps(:,i) .* par.S);
end
r = y;
G = zeros(K,1);
for i=1:K
    G(i) = sum(abs(par.S - x_(i)).^2 .* par.ps(:,i));
end
nu = mean(G);

x_old = ones(K,1);
count = 0;
q = 1/N*ones(N,K);
while norm(x_ - x_old) > 1e-5 && count < par.iters
    count = count + 1;
    x_old = x_;
    z = x_ + H'*r;  % linear estimator
    sigma_sq = N0 + beta*nu;  % error variance 
    for i=1:K
        [x_(i), G(i), q(:,i)] = ...
                denoiser_discrete(z(i), sigma_sq, par.S, par.ps(:,i));
    end
    nu = sum(G) / K;
    r = y - H*x_ + beta*nu/sigma_sq*r;  % onsager corrected residual
end

x_est = zeros(K,1);
for i=1:K
    [~, idx] = max(q(:,i));
    x_est(i) = par.S(idx);
end
