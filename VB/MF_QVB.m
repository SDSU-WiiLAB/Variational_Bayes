% opt_gamma = 'post': MF-QVB
% opt_gamma = anything else: conventional QVB using noise variance = N0
function [x_est, q, r, sigma_r] = MF_QVB(par, H, y, y_l, y_u, delta, N0, opt_gamma)
K = par.K;
M = par.M;
N = par.const_size;
diff = 1;
r = y;
sigma_r = delta^2/6*ones(M, 1);
x_ = zeros(K, 1);
q = 1/N*ones(N,K);
count = 0;
norm_H = sum(H.*conj(H),1)'; %vecnorm(H).^2';
% HH = H.*conj(H);
G = par.P*ones(K, 1);
z = zeros(K, 1);
res = r - H*x_;
while diff > 1e-6 && count < par.iters
    count = count + 1;
    x_old = x_;
    if strcmp(opt_gamma, 'post') || opt_gamma==1
        gamma = M/(res'*res + sum(sigma_r) + sum(norm_H.*G));
    else
        gamma = 1/N0;
    end
    for i = 1:K
        z(i) = x_(i) + H(:,i)'*res./norm_H(i); % linear estimate
        [x_hat, G(i), q(:,i)] = denoiser_discrete(z(i), ...
                1/gamma/norm_H(i), par.S, par.ps);  % denoiser
        res = res + H(:,i)*(x_(i) - x_hat); % update residual
        x_(i) = x_hat; % update nonlinear estimate
    end

    mu = H*x_;
    [r_real, sigma_r_real] = find_r_real(real(mu), sqrt(1/gamma/2), real(y), real(y_l), real(y_u), delta);
    [r_imag, sigma_r_imag] = find_r_real(imag(mu), sqrt(1/gamma/2), imag(y), imag(y_l), imag(y_u), delta);
    r_temp = r_real + 1i*r_imag;
    sigma_r = sigma_r_real + sigma_r_imag;
    res = res - r + r_temp;
    r = r_temp;
    diff = norm(x_ - x_old);
end
x_est = zeros(K,1);
for i=1:K
    [~, idx] = max(q(:,i));
    x_est(i) = par.S(idx);
end
% r.'
end
