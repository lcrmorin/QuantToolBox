function y = drcLogLogistic(x,alpha,beta,y_min,y_max)
    y = 1 + exp(-beta .* (log(x) - log(alpha)));
    y = y_min + (y_max - y_min) ./ y;
end
