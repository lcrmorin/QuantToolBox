function [u1,u2] = rndCopulaNormal2(rho,ns,mtd)
    if nargin == 2
        mtd = 0;
    end
    u = rndnCopula(ns,2,mtd);
    u1 = u(:,1);
    u2 = u(:,2);
    e = ones(rows(rho),cols(rho));
    u2 = cdfn(rho .*u1 + sqrt(1-rho.^2) .* u2);
    u1 = cdfn(u1) .* e;
end
