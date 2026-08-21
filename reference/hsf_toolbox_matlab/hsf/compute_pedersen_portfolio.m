function [w,results,C_x_y] = compute_pedersen_portfolio(mu,r,Sigma,S,sigma_bar,S_bar)

pi_ = mu - r;
inv_Sigma = inv(Sigma);
n = rows(mu);
e = ones(n,1);

% C_x_y

C_1_pi = e'*inv_Sigma*pi_;
C_1_s = e'*inv_Sigma*S;
C_s_pi = S'*inv_Sigma*pi_;
C_s_s = S'*inv_Sigma*S;
C_1_1 = e'*inv_Sigma*e;
C_pi_pi = pi_'*inv_Sigma*pi_;
C_x_y = [C_1_pi; C_s_pi; C_s_s; C_1_s; C_1_1; C_pi_pi];

nIters = max(rows(sigma_bar),rows(S_bar));
all_sigma_bar = sigma_bar .* ones(nIters,1);
all_S_bar = S_bar .* ones(nIters,1);
all_lambda1 = zeros(nIters,1);
all_lambda2 = zeros(nIters,1);
all_w = zeros(n,nIters);
all_pi_w = zeros(nIters,1);
all_sigma_w = zeros(nIters,1);
all_S_w = zeros(nIters,1);
all_SR_w = zeros(nIters,2);

for iter = 1:nIters
    sigma_bar = all_sigma_bar(iter);
    S_bar = all_S_bar(iter);
    lambda2 = (C_1_pi * S_bar - C_s_pi)/(C_s_s - 2*C_1_s * S_bar + C_1_1 * S_bar^2);
    aux = C_pi_pi - ((C_1_pi * S_bar - C_s_pi)^2)/(C_s_s - 2*C_1_s * S_bar + C_1_1 * S_bar^2);
    lambda1 = -1/(2*sigma_bar)*sqrt(aux);
    w = (-1/(2*lambda1))* inv_Sigma*(pi_ + lambda2*(S - S_bar));
    pi_w = w'*pi_;
    sigma_w = sqrt(w'*Sigma*w);
    S_w = w'*S/sum(w);
    SR1_w = (w'*pi_)/sigma_bar;
    SR2_w = C_pi_pi - ((C_1_pi * S_bar - C_s_pi)^2)/(C_s_s - 2*C_1_s * S_bar + C_1_1 * S_bar^2);
    SR2_w = sqrt(SR2_w);
    all_lambda1(iter) = lambda1;
    all_lambda2(iter) = lambda2;
    all_w(:,iter) = w;
    all_pi_w(iter) = pi_w;
    all_sigma_w(iter) = sigma_w;
    all_S_w(iter) = S_w;
    all_SR_w(iter,1) = SR1_w;
    all_SR_w(iter,2) = SR2_w;
end

w = all_w;
w_r = 1 - sumc(w);
results.sigma_bar = all_sigma_bar;
results.S_bar = all_S_bar;
results.lambda1 = all_lambda1;
results.lambda2 = all_lambda2;
results.pi_w = all_pi_w;
results.sigma_w = all_sigma_w;
results.S_w = all_S_w;
results.SR_w = all_SR_w;
results.w_r = w_r;

end
