function C = cdfCopulaHuslerReiss(u1,u2,theta)
    u1tilde = -log(u1);
    u2tilde = -log(u2);
    phi1 = 1./theta + 0.5*theta.*log(u1tilde./u2tilde);
    phi2 = 1./theta + 0.5*theta.*log(u2tilde./u1tilde);
    C = exp((-u1tilde .* cdfn(phi1)) + (-u2tilde .* cdfn(phi2)));
end
