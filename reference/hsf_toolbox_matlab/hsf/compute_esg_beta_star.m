function [beta_star,beta_esg_star] = compute_esg_beta_star(beta,sigma_m,beta_esg,sigma_esg,sigma_tilde)

n = rows(beta);
e = ones(n,1);

sigma_m_sqr = sigma_m ^ 2;
sigma_esg_sqr = sigma_esg ^ 2;
sigma_m_esg = sigma_m * sigma_esg;
sigma_m_esg_sqr = sigma_m_esg ^ 2;

beta_tilde = beta ./ (sigma_tilde .^2);
varphi_m = beta' * beta_tilde;

beta_esg_tilde = beta_esg ./ (sigma_tilde .^2);
varphi_esg = beta_esg' * beta_esg_tilde;

varphi_m_esg = beta' * beta_esg_tilde;

omega0 = 1 + sigma_m_sqr*varphi_m + sigma_esg_sqr*varphi_esg + ...
         sigma_m_esg_sqr*(varphi_m * varphi_esg - varphi_m_esg^2);
omega1 = varphi_esg * (beta_tilde'*e) - varphi_m_esg * (beta_esg_tilde'*e);
omega1 = sigma_m_sqr*((beta_tilde'*e) + sigma_esg_sqr*omega1);
omega2 = varphi_m * (beta_esg_tilde'*e) - varphi_m_esg * (beta_tilde'*e);
omega2 = sigma_esg_sqr*((beta_esg_tilde'*e) + sigma_m_sqr*omega2);

beta_star = omega0/omega1;
beta_esg_star = omega0/omega2;

end
