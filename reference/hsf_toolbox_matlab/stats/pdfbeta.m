function p = pdfbeta(x,a,b)
    beta = gamma(a) .* gamma(b) ./  gamma(a + b);
    p = x.^(a-1) .* (1-x).^(b-1) ./ beta;
end
