function C = cdfCopulaFrank(u1,u2,theta)
    u1 = exp(-theta.*u1) - 1;
    u2 = exp(-theta.*u2) - 1;
    C = -log(1 + u1.*u2./(exp(-theta)-1)) ./ theta;
end
