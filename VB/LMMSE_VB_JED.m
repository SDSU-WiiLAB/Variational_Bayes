% opt_lite: default as 'lite'
% Here R: 3D matrix with R(:,:,i) is the covariance matrix for channel h_i
function [X_est, Q, H_] = LMMSE_VB_JED(par, Yd, Yp, Xp, C, opt_lite)
if strcmp(opt_lite, 'full')
    lite = 0;
else
    lite = 1;
end
K = par.K;
M = par.M;
Td = par.Td;
Tp = par.Tp;
X_ = zeros(K, Td);
% zeta = par.zeta;
Q = zeros(par.const_size, K, Td);
% [H_, Sigma_H, ~, ~] = VB_CE_FB_real(par, Yp, Yp_l, Yp_u, Xp, R, zeta, N0, delta, opt_gamma);
PT = sum(Xp.*conj(Xp), 2);
H_ = Yp*Xp'/diag(PT); %zeros(M, K);

Sigma_H = C;
trace_H = arrayfun(@(i)trace(C(:,:,i)), 1:size(C,3))';
C_inv = pageinv(C);
% R_inv = inv(R);
% Sigma_H = zeros(M, M, K);
% trace_H = zeros(K, 1);
% for i=1:K
%     Sigma_H(:,:,i) = R/par.zeta(i);
%     trace_H(i) = trace(Sigma_H(:,:,i));
% end

diff = 1;
count = 0;
% initialization
G = ones(K, Td);
% Z = zeros(K, Td);
W = zeros(M, M, Td);
Ep = Yp - H_*Xp;
Ed = Yd - H_*X_;
Xd_2 = abs(X_).^2 + G;
T1 = zeros(K, Td);
T2 = zeros(K, Td);
while diff > 5e-5 && count < par.iters
    count = count + 1;
    X_old = X_;
    gamma_p = M*Tp/(norm(Ep, 'fro')^2 + trace_H'*PT);
    
    if lite
        temp = 0;
        for i=1:K
            temp = temp + sum(Xd_2(i,:)) * Sigma_H(:,:,i);
        end
        W = Td * inv(norm(Ed,'fro')^2/M*eye(M) + temp + H_*diag(sum(G, 2))*H_');
%         W = Td * inv(Ed*Ed' + diag(sum(Sigma_Rd, 2)) + ...
%             temp + H_*diag(sum(G, 2))*H_');
    else
        for t=1:Td
            temp = 0;
            for i=1:K
                temp = temp + Xd_2(i,t) * Sigma_H(:,:,i);
            end
            W(:,:,t) = inv(norm(Ed(:,t))^2/M*eye(M) + temp + H_*diag(G(:,t))*H_');
        end
    end
    
    % Update X_
    for i=1:K
        if lite
            T1(i,:) =  H_(:,i)'*W*H_(:,i);
            T2(i,:) = real(T1(i,:) + trace(W*Sigma_H(:,:,i)));
        else
            for t=1:Td
                T1(i,t) = H_(:,i)'*W(:,:,t)*H_(:,i);
                T2(i,t) = real(T1(i,t)) + trace(W(:,:,t)*Sigma_H(:,:,i));
            end
        end
    end
    
    for ii=1:3
        for i=1:K
            for t=1:Td
                if lite
                    Z_tilde = (T1(i,t)*X_(i,t) + H_(:,i)'*W*Ed(:,t))/T2(i,t);
                else
                    Z_tilde = (T1(i,t)*X_(i,t) + H_(:,i)'*W(:,:,t)*Ed(:,t))/T2(i,t);
                end
                Q(:,i,t) = par.ps .* exp(-abs(Z_tilde - par.S).^2*T2(i,t));
                Q(:,i,t) = Q(:,i,t)/sum(Q(:,i,t));
                x_hat = sum(Q(:,i,t) .* par.S);
                G(i,t) = real(sum(Q(:,i,t) .* abs(par.S - x_hat).^2));
                Ed(:,t) = Ed(:,t) + H_(:,i)*(X_(i,t) - x_hat);
                X_(i,t) = x_hat;
            end
        end
    end
    Xd_2 = abs(X_).^2 + G;
    
     % Update H
%     for ii=1:3
    for i=1:K
        if lite
            Gamma_i = gamma_p*PT(i)*eye(M) + sum(Xd_2(i,:)) * W;
            k_i = (eye(M) - Gamma_i\W*sum(G(i,:)))*H_(:,i) + ...
                Gamma_i\(gamma_p*Ep*Xp(i,:)' + W*Ed*X_(i,:)');
        else        
            Gamma_i = gamma_p*PT(i)*eye(M);
            temp1 = 0;
            temp2 = 0;
            for t=1:Td
                Gamma_i = Gamma_i + Xd_2(i,t) * W(:,:,t);
                temp1 = temp1 + G(i,t) * W(:,:,t);
                temp2 = temp2 + W(:,:,t) * Ed(:,t) * conj(X_(i,t));
            end
            k_i = (eye(M) - Gamma_i\temp1)*H_(:,i) + ...
                Gamma_i\(gamma_p*Ep*Xp(i,:)' + temp2);
        end
        Sigma_H(:,:,i) = inv(Gamma_i + C_inv(:,:,i));
        trace_H(i) = real(trace(Sigma_H(:,:,i)));
        H_hat = Sigma_H(:,:,i)*Gamma_i*k_i;
        Ep = Ep + (H_(:,i) - H_hat)*Xp(i,:);
        Ed = Ed + (H_(:,i) - H_hat)*X_(i,:);
        H_(:,i) = H_hat;
    end
%     end        
    % Update zeta
%     for i=1:K
%         zeta(i) = M/real(H_(i,:)'*R_inv*H_(i,:) + trace(R_inv*Sigma_H(:,:,i)));
%     end
    diff = norm(X_old - X_, 'fro')/sqrt(Td);
end
X_est = zeros(K, Td);
for i=1:K
    for t=1:Td
        [~, idx] = max(Q(:,i,t));
        X_est(i,t) = par.S(idx);
    end
end
% count