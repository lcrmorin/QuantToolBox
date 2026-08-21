function y = drcHormetic1(x,alpha,beta,y_min,y_max,gamma_)
    y = 1 + exp(-beta .* (log(x) - log(alpha)));
    y = y_min + (y_max - y_min + gamma_ .* x) ./ y;
end
