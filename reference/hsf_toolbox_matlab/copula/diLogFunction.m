function D = diLogFunction(x)
    n = rows(x);
    fun = @(t) log(t)./(1-t);
    D = integral(fun,x,ones(n,1));
end
