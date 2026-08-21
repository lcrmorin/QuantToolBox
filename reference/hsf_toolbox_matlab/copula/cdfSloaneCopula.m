function C = cdfSloaneCopula(u1,u2,rho)
    u1 = acosh(1./(u1.^2));
    u2 = acosh(1./(u2.^2));
    e = u1 <= u2;
    zeta = e .* u1 + (1-e) .* u2;
    xi = abs(u1-u2);

    C1 = cosh(xi) .* cosh(zeta .* sqrt(1+rho) ) .* cosh(zeta .* sqrt(1-rho) ) ;
    C2 = sinh(xi) .* cosh(zeta .* sqrt(1-rho) ) .* sinh(zeta .* sqrt(1+rho) ) ;
    C3 = sinh(xi) .* cosh(zeta .* sqrt(1+rho) ) .* sinh(zeta .* sqrt(1-rho) ) ;

    C = C1 + 0.5*sqrt(1+rho) .* C2 + 0.5*sqrt(1-rho) .*C3;
    C = 1./sqrt(C);

    C = missrv(real(C),0);
end
