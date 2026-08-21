function u = rndCopulaStudent(rho,nu,ns)
    n = randn(ns,rows(rho));
    chi2 = chi2rnd(nu,ns,1);
    u = (n*chol(rho))./sqrt(chi2./nu);
    u = cdft(u,nu);
end
