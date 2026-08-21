function p = pdfSN(x,xi,omega,eta)
  xc = (x - xi) ./ omega;
  cdf = cdfn(eta .* xc);
  pdf = pdfn(xc) ./ omega;
  p = 2 * pdf .* cdf;
end
