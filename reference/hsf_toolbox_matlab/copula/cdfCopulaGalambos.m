function C = cdfCopulaGalambos(u1,u2,theta)
    u1tilde = -log(u1);
    u2tilde = -log(u2);
    C = u1 .* u2 .* exp( (u1tilde.^(-theta) + u2tilde.^(-theta)).^(-1./theta));
end
