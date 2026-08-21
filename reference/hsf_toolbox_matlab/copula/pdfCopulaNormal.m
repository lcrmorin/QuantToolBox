function p = pdfCopulaNormal(u,rho)

  varsigma = cdfni(u)';
  rho_inv = inv(rho) - eye(cols(u));
  r = rows(u);
  p = zeros(r,1);
  i = 1;
  while i <= r
    p(i) = exp(-0.5 * varsigma(:,i)' * rho_inv * varsigma(:,i));
    i = i + 1;
  end
  p = p / sqrt(det(rho));

end
