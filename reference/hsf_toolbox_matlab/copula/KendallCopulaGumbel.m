function tau = KendallCopulaGumbel(theta)
  theta = missex(theta, theta < 1);
  tau = 1 - 1./theta;
end
