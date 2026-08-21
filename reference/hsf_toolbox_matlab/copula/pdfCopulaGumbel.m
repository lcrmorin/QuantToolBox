function p = pdfCopulaGumbel(u1,u2,theta)
    u1tilde = -log(u1);
    u2tilde = -log(u2);
    w = u1tilde.^theta + u2tilde.^theta;
    p = ((u1tilde .* u2tilde).^(theta-1)) .* ((w.^(1./theta)) + theta - 1) ./ ...
        (w.^(2-1./theta)) ./ (u1 .* u2);
    p = p .* cdfCopulaGumbel(u1,u2,theta);
end
