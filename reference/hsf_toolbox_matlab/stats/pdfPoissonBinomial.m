function [k,pmf] = pdfPoissonBinomial(p,mtd)

if nargin == 1
    mtd = 1;
end

n = rows(p);
k = (0:n)';

if mtd == 1 % FFT

    M = 2^nextpow2(2*n);
    omega = exp(-2i * pi / M);
    A = ones(1, M);
    for j = 1:n
        A = A .* (1 - p(j) + p(j) * omega.^(0:M-1));
    end
    pmf = ifft(A);
    pmf = real(pmf(1:n+1))';

else

    pmf(1) = 1;
    next_pmf = zeros(n+1,1);
    for i = 1:n
        next_pmf(1) = (1 - p(i)) * pmf(1);
        next_pmf(i+1) = p(i) * pmf(i);
        for j = 1:i-1
            next_pmf(j+1) = p(i) * pmf(j) + (1 - p(i)) * pmf(j+1);
        end
        pmf = next_pmf;
    end

end

end
