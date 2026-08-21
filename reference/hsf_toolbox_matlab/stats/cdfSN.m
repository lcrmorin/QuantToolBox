function p = cdfSN(x,xi,omega,eta,mtd)

if nargin == 4
    mtd = 0;
end

if mtd == 1
    xc = (x - xi) ./ omega;
    delta = eta ./ sqrt(1 + eta.^2);
    p = 2*cdfbvn(xc,0,-delta);
else
    e = eta >= 0;
    xc = (x - xi) ./ omega;
    delta = (1 - eta.^2) ./ (1 + eta.^2);
    cdf = cdfbvn(xc,xc,delta);
    p = cdf .* e + (2*cdfn(xc)-cdf) .* (1-e);
end

end
