function p = pdfbvn(x1,x2,mu1,mu2,sigma1,sigma2,rho)
    w = 1 - rho.^2;
    x1 = (x1 - mu1)./sigma1;
    x2 = (x2 - mu2)./sigma2;
    p = x1.^2 - 2 * rho .* x1 .* x2 + x2.^2;
    p = exp(-0.5*p./w)/(2*pi*sigma1.*sigma2.*sqrt(w));
end
