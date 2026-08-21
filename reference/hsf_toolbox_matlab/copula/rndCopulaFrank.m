function [u1,u2] = rndCopulaFrank(theta,ns,v)
    if nargin == 2
        v1 = rand(ns,1);
        v2 = rand(ns,1);
    else
        v1 = v(:,1);
        v2 = v(:,2);
    end
    u1 = v1 .* ones(rows(theta),cols(theta));
    u2 = -log(1 + v2 .* (exp(-theta)-1) ./ ...
        (v2 + (1-v2) .* exp(-theta.*u1)))./theta;
end
