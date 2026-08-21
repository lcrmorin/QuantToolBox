function cdf = cdfmvn(x,Mu,Sigma)
% Returns the cdf of the multivariate normal distribution N(Mu,Sigma)

if (cols(x) == 1) && (rows(x) == rows(Mu))
    x = x';
end

cdf = mvncdf(x,Mu',Sigma);

end
