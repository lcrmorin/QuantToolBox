function cdf = cdfBates(x,n)

cdf = 0;
for k = 0:n
    s = n*x > k;
    cdf = cdf + ((-1)^k) .* nchoosek(n,k) .* ((n*x - k).^n) .* s;
end

cdf = cdf / factorial(n);

cdf(x == 0) = 0;
cdf(x == 1) = 1;

end
