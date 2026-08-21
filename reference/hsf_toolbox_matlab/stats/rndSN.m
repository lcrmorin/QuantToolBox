function p = rndSN(r,c,xi,omega,eta,mtd)

  if mtd == 1
    rho = eta ./ sqrt(1 + eta.^2);
    u0 = randn(r, c);
    u1 =  rho .* u0 + sqrt(1 - rho.^2) .* randn(r, c);
    e = (u0 >= 0);
    z = 2 .* u1 .* e - u1;
  else
    rho = (1 - eta.^2) ./ (1 + eta.^2);
    u0 = randn(r, c);
    u1 =  rho .* u0 + sqrt(1 - rho.^2) .* randn(r, c);
    z = max(u0,u1);
    e = (eta >= 0);
    z = 2 * z .* e - z;
  end

  p = xi + omega .* z;
end
