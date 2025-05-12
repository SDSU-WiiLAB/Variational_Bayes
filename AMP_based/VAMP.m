function [x_est, q] = VAMP(par, H, y, N0)
%% initialiation
K = par.K;

% %% Economy implementation
% [U,Sigma,V] = svd(H, 'econ');
% S_diag = diag(Sigma);
% rnk = nnz(S_diag);
% S_diag = S_diag(1:rnk);
% U = U(:,1:rnk);
% V = V(:,1:rnk);
% r1 = zeros(K, 1);
% y_ = diag(1./S_diag)*U'*y;
% q = zeros(par.const_size, K);
% gamma_1 = 1/N0; %1./sum(abs(par.S - r1(1)).^2 .* par.ps);
% x_ = zeros(K, 1);
% G = zeros(K, 1);
% diff = 1;
% count = 0;
% while diff > 1e-5 && count < par.iters
%     count = count + 1;
%     r1_old = r1;  
%     % nonlinear denoiser
%     for i=1:K
%         q(:,i) = par.ps .* exp(-abs(r1(i)-par.S).^2*gamma_1);
%         q(:,i) = q(:,i)/sum(q(:,i));
%         x_(i) = sum(q(:,i) .* par.S);  % posterior mean
%         G(i) = gamma_1*sum(q(:,i) .* abs(par.S - x_(i)).^2); % error variance
%     end
%     alpha = mean(G);
%     r2 = (x_ - alpha*r1)/(1 - alpha);
%     gamma_2 = min([max([gamma_1*(1 - alpha)/alpha, 1e-6]), 1e6]);
%     
%     % linear estimator
%     d = (S_diag.^2/N0)./(S_diag.^2/N0 + gamma_2);
%     gamma_1 = min([max([gamma_2*mean(d) / (K/rnk - mean(d)), 1e-6]), 1e6]);
%     r1 = r2 + (K/rnk)*V*diag(d/mean(d))*(y_ - V'*r2);
%     
%     diff = norm(r1 - r1_old);
% end
% x_est = zeros(K,1);
% for i=1:par.K
%     [~, idx] = max(q(:,i));
%     x_est(i) = par.S(idx);
% end


%% Symmetric implementation
r1 = zeros(K, 1);
gamma1 = 1/N0; %sum(abs(par.S - r1(1)).^2 .* par.ps);
q = zeros(par.const_size, K);
x1 = zeros(K, 1);
G1 = zeros(K, 1);
x_est = zeros(K, 1);
diff = 1;
count = 0;
while diff > 1e-5 && count < par.iters
    count = count + 1;
    r1_old = r1;
    % denoising
    for i=1:K
        q(:,i) = par.ps(:,i) .* exp(-abs(r1(i)-par.S).^2*gamma1);
        q(:,i) = q(:,i)/sum(q(:,i));
        x1(i) = sum(q(:,i) .* par.S);  % posterior mean
        G1(i) = gamma1*sum(q(:,i) .* abs(par.S - x1(i)).^2); % error variance
    end
    alpha1 = mean(G1); % divergence
    gamma2 = min([max([gamma1*(1/alpha1-1), 1e-6]), 1e6]);
    r2 = (x1 - alpha1*r1)/(1-alpha1);

    % LMMSE
    x2 = ((H'*H)/N0 + gamma2*eye(K))\(H'*y/N0 + gamma2*r2);
    alpha2 = gamma2/K*trace(inv((H'*H)/N0 + gamma2*eye(K)));  % divergence
    gamma1 = min([max([gamma2*(1/alpha2-1), 1e-6]), 1e6]);
    r1 = (x2 - alpha2*r2)/(1-alpha2);
    diff = norm(r1_old - r1);
end
for i=1:par.K
    [~, idx] = max(q(:,i));
    x_est(i) = par.S(idx);
end
