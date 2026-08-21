function y = drcHormetic2(x,alpha,beta,y_min,y_max,gamma_,delta)
    y = 1 + exp(-beta .* (log(x) - log(alpha)));
    y = y_min + (y_max - y_min + gamma_ .* exp(-1./(x.^delta))) ./ y;
end
