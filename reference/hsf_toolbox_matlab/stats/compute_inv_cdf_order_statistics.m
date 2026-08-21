function q = compute_inv_cdf_order_statistics(alpha,x,F_x,n,i_select,mtd)

if isempty(i_select)
    i_select = 1:n;
end

if nargin == 5
    mtd = 1;
end

k = (0:n);

if mtd == 1
    binom_coeffs = factorial(n) ./ (factorial(n - k) .* factorial(k));
    p_binomial = binom_coeffs .* (F_x .^ k) .* ((1 - F_x).^(n-k));
else
    e = ones(rows(x),cols(k));
    p_binomial = binopdf(k .* e,n .* e,F_x .* e);
end

F_i_n = zeros(rows(F_x),n);
for i = 1:n
    F_i_n(:,i) = sum(p_binomial(:,i+1:end),2);
end

F_i_n = F_i_n(:,i_select);

q = NaN(rows(alpha),cols(i_select));
for iter = 1:rows(alpha)
    for k = 1:cols(i_select)
        indx = find(F_i_n(:,k) >= alpha(iter),1);
        if ~isempty(indx)
            q(iter,k) = x(indx);
        end
    end
end

end
