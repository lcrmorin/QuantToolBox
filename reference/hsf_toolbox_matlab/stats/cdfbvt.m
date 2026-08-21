function p = cdfbvt(x,y,rho,nu)

    r = max(max(max(rows(x),rows(y)),rows(rho)),rows(nu));
    c = max(max(max(cols(x),cols(y)),cols(rho)),cols(nu));
    e = ones(r,c);

    x = x .* e;
    y = y .* e;
    rho = rho .* e;
    nu = nu .* e;

    p = zeros(r,c);

    for i = 1:r
        for j = 1:c
            C_matrix = [1 rho(i,j); rho(i,j) 1];
            p(i,j) = mvtcdf([x(i,j) y(i,j)],C_matrix,nu(i,j));
        end
    end

end
