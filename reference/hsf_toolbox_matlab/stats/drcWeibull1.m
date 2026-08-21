function y = drcWeibull1(x,alpha,beta,y_min,y_max)
    y = exp(-exp(beta .* (log(x) - log(alpha))));
    y = y_min + (y_max - y_min) .* y;
end
