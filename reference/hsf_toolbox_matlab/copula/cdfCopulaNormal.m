function cdf = cdfCopulaNormal(u,rho)
    x = cdfni(u);
    cdf = cdfmvn(x,zeros(cols(u),1),rho);
end
