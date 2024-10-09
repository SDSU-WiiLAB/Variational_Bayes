% Stable implementation of LMMSE-VB
% @ Developed by Duy Nguyen, duy.nguyen@sdsu.com
% Please cite the following paper if you use the code
% Duy HN Nguyen, Italo Atzeni, Antti Tölli, A Lee Swindlehurst, "A Variational Bayesian Perspective on Massive MIMO Detection", arXiv preprint arXiv:2205.11649.
% https://arxiv.org/pdf/2205.11649
function [x_est, q, z, noise_var] = LMMSE_VB(par, H, y)
K = par.K;
N = par.const_size;
M = par.M;
% mean_s = sum(par.ps.*par.S);
% x_ = mean_s * ones(K, 1);
x_ = zeros(K,1);
for i=1:K
    x_(i) = sum(par.ps(:,i).*par.S);
end
q = 1/N*ones(N,K);
diff = 1;
count = 0;
G = ones(K, 1);
z = zeros(K, 1);
res = y - H*x_;
noise_var = zeros(K, 1);
while (diff > 1e-5) && count < par.iters
    count = count + 1;
    J = H / (res'*res/M*eye(K) + diag(G)*(H'*H));
    x_old = x_;
    for i=1:K
        noise_var(i) = 1 / real(J(:,i)' * H(:,i));
        z(i) = x_(i) + J(:,i)' * res * noise_var(i);
        [x_hat, G(i), q(:,i)] = denoiser_discrete(z(i), noise_var(i), par.S, par.ps(:,i));
        res = res + H(:,i)*(x_(i) - x_hat);
        x_(i) = x_hat;
    end
    diff = norm(x_ - x_old);
end
x_est = zeros(K, 1);
for i=1:K
    [~, idx] = max(q(:,i));
    x_est(i) = par.S(idx);
end
end
