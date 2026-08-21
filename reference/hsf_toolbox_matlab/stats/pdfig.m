function p = pdfig(x,mu,lambda)
    p = sqrt(lambda./(2*pi*x.^3)) .* exp(-0.5*lambda./(mu.^2) .* (x-mu).^2 ./ x);
end
