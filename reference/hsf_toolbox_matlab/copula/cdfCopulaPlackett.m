function C = cdfCopulaPlackett(u1,u2,theta)
    eta = theta - 1;
    w = 1 + eta .* (u1 + u2);
    C = 0.5 * (w - sqrt(w.^2 - 4 * theta .* eta .* u1 .* u2)) ./ eta;
end
