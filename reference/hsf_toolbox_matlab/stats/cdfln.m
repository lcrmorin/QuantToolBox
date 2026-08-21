function p = cdfln(x,mu,sigma)
y = (log(x) - mu) ./ sigma;
p = cdfn(y);
end
