function cdf = cdfCopulaStudent(u,rho,nu)
    x = tinv(u,nu);
    cdf = mvtcdf(x,rho,nu);
end
