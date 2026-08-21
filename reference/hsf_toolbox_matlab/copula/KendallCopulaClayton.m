function tau = KendallCopulaClayton(theta)
  theta = missex(theta, theta < -1);
  tau = theta ./ (theta + 2);
end
