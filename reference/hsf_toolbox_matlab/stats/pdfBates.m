function pdf = pdfBates(x,n)

pdf = 0;
for k = 0:n
    pdf = pdf + ((-1)^k) .* nchoosek(n,k) .* ((n*x - k).^(n-1)) .* sign(n*x-k);
end

pdf = 0.5 * pdf * n / factorial(n-1);

if n > 1
    pdf(x == 0) = 0;
    pdf(x == 1) = 0;
else
    pdf(x == 0) = 1;
    pdf(x == 1) = 1;
end

end
