function p = momST(xi, omega, eta, nu)

  delta = eta ./ sqrt(1 + eta^2);
  m0 = delta .* sqrt(nu/pi) .* exp(gammaln(0.5*(nu-1)) - gammaln(0.5 * nu));
  mu = xi + omega .* m0;

  sigma = omega .* sqrt(nu ./ (nu - 2) - m0^2);

  gamma1 = m0 .* ( nu .* (3 - delta^2) ./ (nu - 3) -3 .* nu ...
        ./ (nu - 2) + 2 * m0^2) .* (nu ./ (nu - 2) - m0^2)^(-3/2);

  gamma2 = ( 3 * (nu^2) ./ (nu - 2) ./ (nu - 4) - 4 * (m0^2) .* nu .* ...
      (3 - delta^2) ./ (nu - 3) + 6 * (m0^2) .* nu ./ (nu - 2) - 3 * ...
      (m0^4) ) .* (nu ./ (nu - 2) - m0^2)^(-2) - 3.0;

  p = [mu, sigma, gamma1, gamma2];
end
