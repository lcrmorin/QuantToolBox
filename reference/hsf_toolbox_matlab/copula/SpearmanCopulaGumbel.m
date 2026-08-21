function varrho = SpearmanCopulaGumbel(theta)
    n = rows(theta);
    varrho = zeros(n,1);
    for iter = 1:n
        fun = @(u1,u2) cdfCopulaGumbel(u1,u2,theta(iter));
        varrho(iter) = 12 * integral2(fun,0,1,0,1) - 3;
    end
end
