% Stable version of LMMSE-QVB for data detection with few-bit observations
% @ Developed by Duy Nguyen, duy.nguyen@sdsu.com
% Please cite the following paper if you use the code
% 1) Duy H. N. Nguyen, Italo Atzeni, Antti Tölli, A Lee Swindlehurst, "A Variational Bayesian Perspective on Massive MIMO Detection," 
% arXiv preprint arXiv:2205.11649. https://arxiv.org/pdf/2205.11649
% 2) Ly V. Nguyen, A Lee Swindlehurst, Duy H. N. Nguyen, "Variational Bayes for joint channel estimation and data detection in few-bit massive MIMO systems,"
% IEEE Transactions on Signal Processing, July 2024. DOI: 10.1109/TSP.2024.3429009

function [x_est, q, r, sigma_r] = LMMSE_QVB(par, H, y, y_l, y_u, delta)
K = par.K;
M = par.M;
N = par.const_size;
diff = 1;
r = y;
sigma_r = delta^2/12*ones(M, 1);
x_ = zeros(K, 1);
q = 1/N*ones(N,K);
count = 0;
G = ones(K, 1);
z = zeros(K, 1);
res = r - H*x_;
while diff > 1e-6 && count < par.iters
    count = count + 1;
    W = pinv(res'*res/M*eye(M) + diag(sigma_r) + H*diag(G)*H');
    x_old = x_;
    for i=1:length(r)
        mu = r(i) - W(i,:)*res/real(W(i,i));
        [r_real, sigma_r_real] = find_r_real(real(mu), sqrt(1/abs(W(i,i))/2), ...
            real(y(i)), real(y_l(i)), real(y_u(i)), delta);
        [r_imag, sigma_r_imag] = find_r_real(imag(mu), sqrt(1/abs(W(i,i))/2), ...
            imag(y(i)), imag(y_l(i)), imag(y_u(i)), delta);
        r_temp = r_real + 1i*r_imag;
        sigma_r(i) = sigma_r_real + sigma_r_imag;
        res(i) = res(i) - r(i) + r_temp;
        r(i) = r_temp;
    end
    for i = 1:K
        temp = abs(H(:,i)'*W*H(:,i));
        z(i) = x_(i) + H(:,i)'*W*res/temp;
        [x_hat, G(i), q(:,i)] = denoiser_discrete(z(i), 1/temp, par.S, par.ps);
        res = res + H(:,i)*(x_(i) - x_hat);
        x_(i) = x_hat;
    end
    diff = norm(x_ - x_old);
end
x_est = zeros(K,1);
for i=1:K
    [~, idx] = max(q(:,i));
    x_est(i) = par.S(idx);
end
end
