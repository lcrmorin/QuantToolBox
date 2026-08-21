function u2 = contourCopulaFrank(u1,alpha,theta)
    u2 = -log( 1 + (exp(-alpha.*theta)-1) .* (exp(-theta)-1) ./ ...
        (exp(-theta.*u1) - 1) ) ./ theta;
    cnd = abs(imag(u2)) > 0.01;
    u2 = real(missex(u2,cnd));
    cnd = u2 < 0 | u2 > 1;
    u2 = missex(u2,cnd);
end
