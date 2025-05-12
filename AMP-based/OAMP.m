function [x_est, q, z, sigma_sq] = OAMP(par, H, y, N0, opt_denoiser)
% initialiation
[M, K] = size(H);
N = par.const_size;
x_ = zeros(K,1);
for i=1:K
    x_(i) = sum(par.ps(:,i) .*par.S);
end
q = 1/N*ones(N,K);
eps = 1e-3;
diff = 1;
count = 0;
tau = max((norm(y)^2 - M*N0)/trace(H*H'), eps);
G = zeros(par.K,1);
A_ZF = pinv(H);
A_ZF = K/trace(A_ZF*H)*A_ZF;
A_MF = K/trace(H'*H)*H';
while diff > 1e-5 && count < par.iters
    count = count + 1;
    x_old = x_;
    if strcmp(opt_denoiser,'MMSE')
        A = tau*(tau*(H'*H) + N0*eye(K))\H';
        A = K/trace(A*H)*A;
    elseif strcmp(opt_denoiser,'MF')
        A = A_MF;
    elseif strcmp(opt_denoiser,'ZF')
        A = A_ZF;
    end
    z = x_old + A*(y - H*x_old);   % linear estimator
    sigma_sq = (norm(eye(K) - A*H, 'fro')^2*tau + norm(A, 'fro')^2*N0)/K;
    for i=1:K
        [x_(i), G(i), q(:,i)] = denoiser_discrete(z(i), sigma_sq, par.S, par.ps(:,i));
    end
    temp = sum(G)/K;
    x_ = sigma_sq/(sigma_sq - temp) * (x_ - temp/sigma_sq*z);
    tau = max([(norm(y - H*x_)^2 - par.M*N0)/trace(H*H'), eps]);
    diff = norm(x_ - x_old);
end
x_est = zeros(K,1);
for i=1:K
    [~, idx] = max(q(:,i));
    x_est(i) = par.S(idx);
end
end