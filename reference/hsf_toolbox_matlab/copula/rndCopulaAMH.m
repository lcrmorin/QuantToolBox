function [u1,u2] = rndCopulaAMH(theta,ns,v)
    if nargin == 2
        v1 = rand(ns,1);
        v2 = rand(ns,1);
    else
        v1 = v(:,1);
        v2 = v(:,2);
    end
    u1 = v1;
    [a,b,c,x1,x2] = invcdfConditionalCopulaAMH(u1,v2,theta);
    u2 = x1;
end

function [a,b,c,x1,x2] = invcdfConditionalCopulaAMH(u1,v,theta)
    a = v .* (theta.^2) .* (1 - u1).^2 - theta;
    b = 2 .* theta .* v .* (1 - theta + theta .* u1) .* (1-u1) - ...
        (1 - theta);
    c = v .* (1 - theta + theta .* u1).^2;
    delta = b.^2 - 4 .* a .* c;

    x1 = (-b - sqrt(delta))./(2 .* a);
    x2 = (-b + sqrt(delta))./(2 .* a);
end
