function C = cdfCopulaCubic(u1,u2,theta)
    C = u1.*u2 + theta .* u1.*(u1-1).*(2*u1-1) .* u2.*(u2-1).*(2*u2-1);
end
