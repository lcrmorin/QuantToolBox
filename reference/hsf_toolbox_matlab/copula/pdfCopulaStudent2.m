function pdf = pdfCopulaStudent2(u1,u2,rho,nu)
    varsigma1 = tinv(u1,nu);
    varsigma2 = tinv(u2,nu);
    varsigma1_sqr = varsigma1 .^2;
    varsigma2_sqr = varsigma2 .^2;
    rho_sqr = rho .^2;
    qF = varsigma1_sqr + varsigma2_sqr - 2 * rho .* varsigma1 .* varsigma2;
    pdf = 0.5 * (1 + qF./(1-rho_sqr)./nu).^(-(nu+2)/2) ./ ...
        ( (1 + varsigma1_sqr./nu).*(1 + varsigma2_sqr./nu) ).^(-(nu+1)/2) .* ...
        (gamma(nu/2).^2) .* nu ./ sqrt(1-rho_sqr) ./ gamma((nu+1)/2).^2;
    pdf(isnan(pdf)) = 0;
end
