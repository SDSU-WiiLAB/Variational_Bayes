% opt_lite: default as 'lite'
% opt_lite = None: conventional QVB
function [X_est, Q, H_] = MF_QVB_JED(par, Yd, Yd_l, Yd_u, Yp, Yp_l, Yp_u, Xp, C, delta, N0, opt_lite)
K = par.K;
M = par.M;
Td = par.Td;
Tp = par.Tp;
X_ = zeros(K, Td);
Q = zeros(par.const_size, K, Td);
% [H_, Sigma_H, ~, ~] = VB_CE_FB_real(par, Yp, Yp_l, Yp_u, Xp, R, zeta, N0, delta, opt_gamma);
PT = sum(Xp.*conj(Xp), 2);
H_ = Yp*Xp'/diag(PT); %zeros(M, K);
simple = 1;
for i = 1:par.K
    if ~isdiag(C(:,:,i)) || all(abs(diag(C(:,:,i)) - C(1,1,i)) > 1e-8)
        simple = 0;
        break;
    end
end
if simple == 1
    sigma_h = squeeze(C(1,1,:));
    trace_H = M * sigma_h;
else
    Sigma_H = C;
    trace_H = arrayfun(@(i)trace(C(:,:,i)), 1:size(C,3))';
    C_inv = pageinv(C);
end

% if isdiag(R) && all(abs(diag(R) - R(1,1)) < 1e-8)
%     simple = 1;
%     sigma_h = R(1,1)./zeta;%  squeeze(Sigma_H(1,1,:));%
%     trace_H = M*sigma_h;
% else
%     simple = 0;
%     R_inv = inv(R);
%     Sigma_H = zeros(M, M, K);
%     trace_H = zeros(K, 1);
%     for i=1:K
%         Sigma_H(:,:,i) = R/par.zeta(i);
%         trace_H(i) = trace(Sigma_H(:,:,i));
%     end
% end
diff = 1;
count = 0;

% initialization
Rp = Yp;
Rd = Yd;
Sigma_Rp = delta^2/6*ones(M, Tp);
Sigma_Rd = delta^2/6*ones(M, Td);
G = ones(K, Td);
Z = zeros(K, Td);
norm_H = sum(H_.*conj(H_),1)';
Ep = Rp - H_*Xp;
Ed = Rd - H_*X_;
Xd_2 = abs(X_).^2 + G;
while diff > 1e-6 && count < par.iters
    count = count + 1;
    X_old = X_;
    
    if strcmp(opt_lite, 'full')
        gamma_p = M*Tp/(norm(Ep, 'fro')^2 + sum(sum(Sigma_Rp)) + trace_H'*PT);
        gamma_d = M./(sum(Ed.*conj(Ed), 1)' + sum(Sigma_Rd, 1)' + ...
                Xd_2'*trace_H + G'*norm_H);
    elseif strcmp(opt_lite, 'None')
        gamma_p = 1/N0;
        gamma_d = 1/N0 * ones(Td, 1);
    else
        gamma_p = M*Tp/(norm(Ep, 'fro')^2 + sum(sum(Sigma_Rp)) + trace_H'*PT);
        gamma_d = M*Td/(norm(Ed, 'fro')^2 + sum(sum(Sigma_Rd)) + ...
                sum(Xd_2'*trace_H) + sum(G'*norm_H)) * ones(Td, 1);
    end
       
     % Update X_
    Sigma_tilde = 1./((norm_H + trace_H)*gamma_d');
    for ii=1:3
    for i=1:K
        Z(i,:) = X_(i,:) + H_(:,i)'*Ed/norm_H(i);
        for t=1:Td
            Z_tilde = Z(i,t)*norm_H(i)/(norm_H(i) + trace_H(i));
            % denoise for x
            Q(:,i,t) = par.ps .* exp(-abs(Z_tilde - par.S).^2/Sigma_tilde(i,t));
            Q(:,i,t) = Q(:,i,t)/sum(Q(:,i,t));
            x_hat = sum(Q(:,i,t) .* par.S);
            G(i,t) = real(sum(Q(:,i,t) .* abs(par.S - x_hat).^2));
            Ed(:,t) = Ed(:,t) + H_(:,i)*(X_(i,t) - x_hat);
            X_(i,t) = x_hat;
        end
    end
    end
    Xd_2 = abs(X_).^2 + G;
    
    % Update Rp and Rd
    mu = Rp - Ep;
    [r_real, sigma_r_real] = find_r_real(real(mu), sqrt(1/gamma_p/2), ...
        real(Yp), real(Yp_l), real(Yp_u), delta);
    [r_imag, sigma_r_imag] = find_r_real(imag(mu), sqrt(1/gamma_p/2), ...
        imag(Yp), imag(Yp_l), imag(Yp_u), delta);
    temp = r_real + 1i*r_imag;
    Sigma_Rp = sigma_r_real + sigma_r_imag;
    Ep = Ep - Rp + temp;
    Rp = temp;
    
    for t=1:Td
        mu = Rd(:,t) - Ed(:,t); %H_*X_(:,t);
        [r_real, sigma_r_real] = find_r_real(real(mu), sqrt(1/gamma_d(t)/2), ...
            real(Yd(:,t)), real(Yd_l(:,t)), real(Yd_u(:,t)), delta);
        [r_imag, sigma_r_imag] = find_r_real(imag(mu), sqrt(1/gamma_d(t)/2), ...
            imag(Yd(:,t)), imag(Yd_l(:,t)), imag(Yd_u(:,t)), delta);
        temp = r_real + 1i*r_imag;
        Sigma_Rd(:,t) = sigma_r_real + sigma_r_imag;
        Ed(:,t) = Ed(:,t) - Rd(:,t) + temp;
        Rd(:,t) = temp;
    end
    
     % Update H_
%     for ii=1:3
    for i=1:K
        gamma_i = PT(i)*gamma_p + Xd_2(i,:)*gamma_d;
        k_i = (1 - G(i,:)*gamma_d/gamma_i)*H_(:,i) + ...
            (gamma_p*Ep*Xp(i,:)' + Ed*(X_(i,:)'.*gamma_d))/gamma_i;
        if simple
            sigma_h(i) = 1/(gamma_i + 1/C(1,1,i));
            H_hat = gamma_i*sigma_h(i)*k_i;
            trace_H(i) = M*sigma_h(i);
        else
            Sigma_H(:,:,i) = inv(gamma_i*eye(M) + C_inv);
            H_hat = gamma_i*Sigma_H(:,:,i)*k_i;
            trace_H(i) = real(trace(Sigma_H(:,:,i)));
        end
        Ep = Ep + (H_(:,i) - H_hat)*Xp(i,:);
        Ed = Ed + (H_(:,i) - H_hat)*X_(i,:);
        H_(:,i) = H_hat;
    end
%     end    
    norm_H = sum(H_.*conj(H_), 1)';
    
    % Update zeta
%     for i=1:K
%         if simple
%             zeta(i) = M/(H_(:,i)'*H_(:,i)/R(1,1) + M*sigma_h(i)/R(1,1));
%         else
%             zeta(i) = M/real(H_(i,:)'*R_inv*H_(i,:) + trace(R_inv*Sigma_H(:,:,i)));
%         end
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