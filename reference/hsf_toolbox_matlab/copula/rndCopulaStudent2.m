function [u1,u2] = rndCopulaStudent2(rho,nu,ns)
    n = randn(ns,2);
    chi2 = chi2rnd(nu,ns,1);
    n1 = n(:,1) .* ones(rows(rho),cols(rho));
    n2 = rho .* n(:,1) + sqrt(1-rho^2) .* n(:,2);
    chi2 = sqrt(chi2./nu);
    u1 = cdft(n1./chi2,nu);
    u2 = cdft(n2./chi2,nu);
end
