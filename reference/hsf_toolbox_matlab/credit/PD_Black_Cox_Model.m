function [PD_tau,S_tau,d1,d2,varphi] = PD_Black_Cox_Model(A0,mu_A,sigma_A,B,tau)
    sigma_tau = sigma_A .* sqrt(tau);
    sigma2_A = sigma_A .* sigma_A;
    nu_A = mu_A - 0.5 * sigma2_A;
    varphi = (B./A0) .^ (2 * nu_A ./ sigma2_A);

    d1 = (log(A0) - log(B) + mu_A .* tau) ./ sigma_tau - 0.5 * sigma_tau;
    d2 = (log(B) - log(A0) + mu_A .* tau) ./ sigma_tau - 0.5 * sigma_tau;

    S_tau = normcdf(d1) - varphi .* normcdf(d2);
    PD_tau  = 1 - S_tau;
end
