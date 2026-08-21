function p = pdfCopulaCubic(u1,u2,theta)
    p = 1 + theta .* (6*u1.^2-6*u1+1) .* (6*u2.^2-6*u2+1);
end
