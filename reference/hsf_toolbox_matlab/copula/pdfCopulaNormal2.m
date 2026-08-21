function p = pdfCopulaNormal2(u1,u2,rho)

    x1 = cdfni(u1);
    x2 = cdfni(u2);
    x1_sqr = x1.^2;
    x2_sqr = x2.^2;
    rho_sqr = rho.^2;

    p = exp(-0.5 ./ (1-rho_sqr) .* (x1_sqr + x2_sqr - 2*rho.*x1.*x2)) .* ...
        exp(0.5 * (x1_sqr + x2_sqr)) ./ sqrt(1-rho_sqr);
end
