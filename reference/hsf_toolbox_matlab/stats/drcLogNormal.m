function y = drcLogNormal(x,alpha,beta,y_min,y_max)
    y = cdfn(beta .* (log(x) - log(alpha)));
    y = y_min + (y_max - y_min) .* y;
end
