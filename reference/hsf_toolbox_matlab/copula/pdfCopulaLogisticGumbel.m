function p = pdfCopulaLogisticGumbel(u1,u2)
    p = 2 * u1 .* u2 ./ (u1 + u2 - u1 .* u2).^3;
end
