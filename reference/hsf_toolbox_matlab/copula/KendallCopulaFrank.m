function tau = KendallCopulaFrank(theta)
    n = rows(theta);
    tau = zeros(n,1);
    for iter = 1:n
        tau(iter) = 1 - 4 .* (1-DebyeFunction(theta(iter),1)) ./ theta(iter);
    end
end
