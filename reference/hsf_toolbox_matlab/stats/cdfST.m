function p = cdfST(x, xi, omega, eta, nu, mtd)

if nargin == 5
    mtd = 0;
end

  if mtd == 1
    xc = (x - xi) ./ omega;
    delta = eta ./ sqrt(1 + eta .^2);
    cdf = 2 .* cdfbvt(xc, 0, -delta, nu);
  else
    e = eta >= 0;
    eta = max(abs(eta), 1e-8);
    xc = (x - xi) ./ omega;
    delta = (1 - eta .^2) ./ (1 + eta .^2);
    cdf = cdfbvt(xc, xc, delta, nu);
    cdf = cdf .* e + (2 .* cdft(xc, nu) - cdf) .* (1 - e);
  end

  p = cdf;
end
