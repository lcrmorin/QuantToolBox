function [x,results] = compute_esg_minimum_variance(beta,sigma_m,beta_esg,sigma_esg,sigma_tilde,e)

n = rows(beta);
D = diag(sigma_tilde .^ 2);
sigma_m_sqr = sigma_m ^ 2;
sigma_esg_sqr = sigma_esg ^ 2;
Sigma = beta*beta'*sigma_m_sqr + beta_esg*beta_esg'*sigma_esg_sqr + D;

if nargin == 5
    e = ones(n,1);
end

beta = e .* beta;
beta_esg = e .* beta_esg;
[beta_star,beta_esg_star] = compute_esg_beta_star(beta,sigma_m,beta_esg,sigma_esg,sigma_tilde);
x = 1 ./ (sigma_tilde .^2) .* (1 - beta/beta_star - beta_esg/beta_esg_star);
x = e .* x;
x = x / sumc(x);
sigma_x = sqrt(x'*Sigma*x);

results.Sigma = Sigma;
results.sigma_x = sigma_x;
results.beta_star = beta_star;
results.beta_esg_star = beta_esg_star;

Sigma_tilde = Sigma(logical(e),logical(e));
n_tilde = rows(Sigma_tilde);
e_tilde = ones(n_tilde,1);
x_tilde = (inv(Sigma_tilde)*e_tilde) / (e_tilde'*inv(Sigma_tilde)*e_tilde);
results.Sigma_tilde = Sigma_tilde;
results.x_tilde = x_tilde;

end
