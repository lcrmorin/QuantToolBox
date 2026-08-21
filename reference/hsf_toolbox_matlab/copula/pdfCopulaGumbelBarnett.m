function p = pdfCopulaGumbelBarnett(u1,u2,theta)
    p = (1 - theta - theta .* (log(u1) + log(u2)) + theta.^2 .* log(u1) .* log(u2)) .* ...
        exp(-theta .* log(u1) .* log(u2));
end
