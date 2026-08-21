function C = cdfCopulaMarshallOlkin(u1,u2,theta1,theta2)
    C = u1.^(1-theta1) .* u2.^(1-theta2) .* ...
        (u1.^theta1 + (u2.^theta2 - u1.^theta1) .* (u1.^theta1 > u2.^theta2));
end
