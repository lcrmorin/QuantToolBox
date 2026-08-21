function C = cdfCopulaClayton(u1,u2,theta)
    theta = missex(theta, theta < -1);
    C = max(u1.^(-theta) + u2.^(-theta) - 1,0).^(-1./theta);
end
