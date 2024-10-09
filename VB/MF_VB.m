% Stable version of MF-VB
function [x_est, q, z, noise_var] = MF_VB(par, H, y)
K = par.K;
N = par.const_size;
x_ = zeros(K,1);
for i=1:K
    x_(i) = sum(par.ps(:,i) .* par.S);
end
G = zeros(K,1);
for i=1:K
    G(i) = sum(abs(par.S - x_(i)).^2 .* par.ps(:,i));
end

q = 1/N*ones(N,K);
norm_H = vecnorm(H).^2';
res = y - H * x_;
z = zeros(K, 1);
diff = 1;
count = 0;
while (diff > 1e-5) && count < par.iters
    count = count + 1;
    x_old = x_;
    gamma = (par.M)/(res'*res + sum(norm_H.*G)); % VBEM for estimate the noise
    for i=1:K
        z(i) = x_(i) + H(:,i)'*res./norm_H(i); % linear estimate
        [x_hat, G(i), q(:,i)] = denoiser_discrete(z(i), ...
            1/gamma/norm_H(i), par.S, par.ps(:,i));
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
noise_var = 1/gamma./norm_H;
end
