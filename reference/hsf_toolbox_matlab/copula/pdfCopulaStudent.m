function pdf = pdfCopulaStudent(u,rho,nu)
    n = cols(u);
    rho_inv = inv(rho);
    rho_det = det(rho);
    varsigma = tinv(u,nu)';
    r = rows(u);
    pdf = zeros(r,1);
    for i = 1:r
    pdf(i) = (1 + varsigma(:,i)' * rho_inv * varsigma(:,i) / nu)^(-(nu+n)/2);
    end
    pdf = pdf ./ prodc( (1+(varsigma.^2)/nu).^(-(nu+1)/2) ) .* gamma( (nu+n)/2 ) .* ...
        (gamma( nu/2 ).^(n-1)) ./ (gamma( (nu+1)/2 ).^n) ./ sqrt(rho_det);
    pdf(isnan(pdf)) = 0;
end
