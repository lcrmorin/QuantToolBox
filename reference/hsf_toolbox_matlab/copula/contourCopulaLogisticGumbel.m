function u2 = contourCopulaLogisticGumbel(u1,alpha)
    u1 = missex(u1,u1 < alpha);
    u2 = alpha .* u1 ./ (u1 + alpha .* u1 - alpha);
end
