function p = pdfCopulaFrank(u1,u2,theta)
    eta = 1-exp(-theta);
    v1 = 1-exp(-theta.*u1);
    v2 = 1-exp(-theta.*u2);
    p = exp(-theta.*(u1+u2)) .* theta .* eta ./ ((eta - v1 .* v2).^2);
end
