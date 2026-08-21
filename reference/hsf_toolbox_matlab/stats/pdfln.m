function p = pdfln(x,mu,sigma)
y = (log(x) - mu) ./ sigma;
p = 1./(x .* sqrt(2 * pi) .* sigma) .* exp(-0.5 .* (y.^2));
end
