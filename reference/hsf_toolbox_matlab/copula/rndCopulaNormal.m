function u = rndCopulaNormal(rho,ns,mtd)
    if nargin == 2
        mtd = 0;
    end
    u = rndnCopula(ns,rows(rho),mtd);
    u = cdfn(u*chol(rho));
end
