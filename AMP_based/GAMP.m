function [x_est, q, r, tau_r] = GAMP(par, H, y, N0)
% initialiation
% option = 0 for detection with discrete input
% Default: signal recovery with Gaussian-Bernoulli prior
K = par.K;
M = par.M;
N = par.const_size;
x_ = zeros(K,1);
for i=1:K
    x_(i) = sum(par.ps(:,i) .* par.S);
end
tau_x = zeros(K,1);
for i=1:K
    tau_x(i) = sum(abs(par.S - x_(i)).^2 .* par.ps(:,i));
end

x_old = ones(K,1);
count = 0;
q = 1/N*ones(N,K);
HH = H.*conj(H);
s = zeros(M, 1);

while norm(x_ - x_old) > 1e-5 && count < par.iters
    count = count + 1;
    x_old = x_;
    
    % Output linear step
    tau_p = HH*tau_x;
    p = H*x_ - tau_p.*s;
    
    % Output nonlinear step
    [s, tau_s] = denoiser_Gaussian(p, tau_p, y, N0); 
    s = (s - p)./tau_p;
    tau_s = (tau_p - tau_s)./tau_p.^2;

    % Input linear step
    tau_r = 1./(HH'*tau_s);
    r = x_ + tau_r.*(H'*s);

    % Input nonlinear step
    [x_(i), tau_x(i), q(:,i)] = ...
                denoiser_discrete(r(i), tau_r(i), par.S, par.ps(i,:));
%     norm(x_ - x_old)
end

x_est = zeros(K,1);
for i=1:K
    [~, idx] = max(q(:,i));
    x_est(i) = par.S(idx);
end
end
