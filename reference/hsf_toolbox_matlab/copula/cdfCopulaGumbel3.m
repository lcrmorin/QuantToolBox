function C = cdfCopulaGumbel3(u1,u2,u3,theta1,theta2)
    u1 = -log(u1);
    u2 = -log(u2);
    u3 = -log(u3);
    C = u1.^theta2 + u2.^theta2;
    C = exp(-(C.^(theta1./theta2) + u3.^theta1).^(1./theta1));
end
