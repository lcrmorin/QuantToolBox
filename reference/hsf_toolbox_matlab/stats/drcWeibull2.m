function y = drcWeibull2(x,alpha,beta,y_min,y_max)
    y = 1 - exp(-exp(beta .* (log(x) - log(alpha))));
    y = y_min + (y_max - y_min) .* y;
end
