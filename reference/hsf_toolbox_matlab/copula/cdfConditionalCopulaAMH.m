function C = cdfConditionalCopulaAMH(u1,u2,theta)
    N = (1 - theta) .* u2 + theta .* (u2 .^2);
    D = (1 - theta .* (1-u1) .* (1-u2) ).^2;
    C = N ./ D;
end
