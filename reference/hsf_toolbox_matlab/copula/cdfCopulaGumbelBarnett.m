function C = cdfCopulaGumbelBarnett(u1,u2,theta)
    C = u1 .* u2 .* exp(-theta .* log(u1) .* log(u2));
end
