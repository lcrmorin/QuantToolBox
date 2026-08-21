function p = rndST(r, c, xi, omega, eta, nu)

  n = rndSN(r, c, 0, omega, eta, 1);
  u = rand(r, c);
  chi = chi2inv(u, nu .* ones(r, c));

  p = xi + n./sqrt(chi ./ nu);
end
