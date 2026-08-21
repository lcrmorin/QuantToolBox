function [u1,u2] = rndCopula2(cndCopula2,ns,v)
    if nargin == 2
        v1 = rand(ns,1);
        v2 = rand(ns,1);
    else
        v1 = v(:,1);
        v2 = v(:,2);
    end
    u1 = v1;
    u2 = rndCopula2FindZero(cndCopula2,v1,v2);
end

function c = rndCopula2FindZero(cndCopula2,v1,v2)
    global COPULA_macheps

    f = @(u) cndCopula2(v1,u) - v2;
    nobs = rows(v1);

    a = zeros(nobs,1) + COPULA_macheps;
    b = ones(nobs,1) - COPULA_macheps;
    ya = f(a);
    yb = f(b);
    nobs = rows(ya);
    a = a .* ones(nobs,1);
    b = b .* ones(nobs,1);

    cnd = (ya < 0) & (yb > 0);
    if sum(cnd,1) == 0
        c = NaN(nobs,1);
        return
    end

    if sum(cnd,1) == nobs

        while max(abs(a-b)) > COPULA_macheps
            c = (a+b)/2;
            yc = f(c);
            cnd1 = yc < 0;
            cnd2 = 1 - cnd1;
            a = cnd1 .* c + cnd2 .* a;
            b = cnd1 .* b + cnd2 .* c;
        end

    else

        s = delif(seqa(1,1,nobs),cnd);
        diff = selif(a-b,cnd);
        missing = NaN(nobs-sumc(cnd),1);

        while max(abs(diff)) > COPULA_macheps
            c = (a+b)/2;
            c(s) = missing;
            yc = f(c);
            cnd1 = yc < 0;
            cnd2 = 1 - cnd1;
            a = cnd1 .* c + cnd2 .* a;
            b = cnd1 .* b + cnd2 .* c;
            diff = selif(a-b,cnd);
        end

    end

    c = (a+b)/2;
end
