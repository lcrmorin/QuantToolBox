function p = momSN(xi,omega,eta)

  delta = eta ./ sqrt(1 + eta^2);
  m0 = delta .* sqrt(2/pi);
  mu = xi + omega .* m0;
  sigma = omega .* sqrt(1 - m0^2);
  gamma1 = (4 - pi) / 2 * (m0^3) ./ ((1 - m0^2)^(3/2));
  gamma2 = 2 * (pi - 3) * (m0^4) ./ ((1 - m0^2)^2);

  p = [mu,sigma,gamma1,gamma2];
end
