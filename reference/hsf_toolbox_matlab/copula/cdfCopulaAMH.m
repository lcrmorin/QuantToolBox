function C = cdfCopulaAMH(u1,u2,theta)
    C = u1 .* u2 ./ (1 - theta .* (1-u1) .* (1-u2));
end
