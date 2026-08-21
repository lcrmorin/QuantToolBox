function y = rndCopulaEmpiricalQuantile(u,x)
    n = cols(u);
    ns = rows(u);
    y = zeros(ns,n);
    x = x .* ones(1,n);

    for i = 1:n
        y(:,i) = rndCopulaEmpiricalQuantileAux(x(:,i),u(:,i));
    end
end

function r = rndCopulaEmpiricalQuantileAux(x,e)
    n = rows(x);
    w = n * e;
    wt = floor(w);
    f = w - wt;
    z = sort(x,1);
    z = [z(1); z; z(n)];
    wt = wt + 1;
    r = z(wt,:) + f .* (z(wt+1,:) - z(wt,:));
end
