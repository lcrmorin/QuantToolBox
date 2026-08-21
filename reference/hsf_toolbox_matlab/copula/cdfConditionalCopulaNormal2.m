function C = cdfConditionalCopulaNormal2(u1,u2,rho)
    C = cdfn((cdfni(u1) - rho.*cdfni(u2)) ./ sqrt(1-rho.^2));
end
