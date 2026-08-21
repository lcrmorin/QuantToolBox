function pdf = pdfCopulaClayton(u1,u2,theta)
    theta = missex(theta, theta < -1);
    pdf = (1 + theta) .* ((u1 .* u2).^(-theta-1)) .* ...
        (max(u1.^(-theta) + u2.^(-theta) - 1,0) .^ (-(2*theta+1)./theta));
    pdf(isnan(pdf)) = 0;
end
