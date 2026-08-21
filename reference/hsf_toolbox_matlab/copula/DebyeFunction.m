function D = DebyeFunction(x,k)
    n = rows(x);
    e = x >= 0;
    x = abs(x);
    fun = @(t) (t.^k) ./ (exp(t) - 1);
    D = (k./(x.^k)) .* integral(fun,zeros(n,1),x);
    D = e .* D + (1-e) .* (D + k .* x ./ (1 + k));
end
