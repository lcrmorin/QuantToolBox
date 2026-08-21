function C = cdfCopulaNormal2(u1,u2,rho)
    C = cdfbvn(cdfni(u1),cdfni(u2),rho);
end
