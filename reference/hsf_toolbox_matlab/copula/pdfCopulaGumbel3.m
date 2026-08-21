function p = pdfCopulaGumbel3(u1,u2,u3,theta1,theta2)
    p = 1./(u1.*u2.*u3);
    u1 = -log(u1);
    u2 = -log(u2);
    u3 = -log(u3);
    A = u1.^theta2 + u2.^theta2;
    B = A.^(theta1./theta2) + u3.^theta1;
    beta1 = 1./theta1;
%    beta2 = 1./theta2;
    beta12 = theta1./theta2;
    p = p .* exp(-(B.^(1./theta1)));
    p = p .* (u1.*u2).^(theta2-1) .* u3.^(theta1-1);
    p = p .* A.^(beta12-2) .* B.^(beta1-2);
    p = p .* (A.^beta12 .* B.^(2./theta1-1) + ...
                 (theta2-theta1) .* B.^beta1 + ...
                 (theta1-1) .* A.^beta12 .* B.^(1./theta1-1) + ...
                 (theta1-1) .* (2*theta1-1) .* B.^(-1) .* A.^beta12 + ...
                 (theta1-1) .* (theta2 - theta1) );
end
