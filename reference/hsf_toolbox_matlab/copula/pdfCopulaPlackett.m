function p = pdfCopulaPlackett(u1,u2,theta)
    eta = theta - 1;
    w = 1 + eta .* (u1 + u2);
    p = theta .* (w - 2 * eta .* u1 .* u2) ./ ...
        ((w.^2 - 4 * theta .* eta .* u1 .* u2) .^ 1.5);
end
