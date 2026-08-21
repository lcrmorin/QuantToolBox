function [u1,u2] = rndCopulaGumbel(theta,ns,v)
    if nargin == 2
        v = rand(ns,2);
    end

    cndCopula2 = @(u1,u2) cndCopulaGumbel(u1,u2,theta);
    [u1,u2] = rndCopula2(cndCopula2,ns,v);
end

function u = cndCopulaGumbel(u1,u2,theta)
    u1tilde = -log(u1);
    u2tilde = -log(u2);
    w = u1tilde.^theta + u2tilde.^theta;
    beta = 1./theta;
    u = exp(-(w.^beta)) .* (1 + (u2tilde./u1tilde).^theta ).^(beta-1) ./ u1;
end
