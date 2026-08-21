function p = pdfST(x, xi, omega, eta, nu)
  xc = (x - xi) ./ omega;
  cdf = cdft(eta .* xc .* sqrt((nu + 1) ./ (xc.^2 + nu)), nu + 1);
  pdf = tpdf(xc,nu) ./ omega;
  p = 2 * pdf .* cdf;
end
