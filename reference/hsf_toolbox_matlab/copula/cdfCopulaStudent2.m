function cdf = cdfCopulaStudent2(u1,u2,rho,nu)
    r = rows(u1);
    if cols(u2) == 1
        u = [u1 u2];
        c = max(rows(rho),rows(nu));
        rho = rho .* ones(c,1);
        nu = nu .* ones(c,1);
        cdf = zeros(r,c);
        for iter = 1:c
            rho_matrix = xpnd([1; rho(iter); 1]);
            cdf(:,iter) = cdfCopulaStudent(u,rho_matrix,nu(iter));
        end
    else
        c = cols(u2);
        u1 = u1 .* ones(r,c);
        u2 = u2 .* ones(r,c);
        cdf = zeros(r,c);
        rho_matrix = xpnd([1; rho; 1]);
        for iter = 1:c
            u = [u1(:,iter) u2(:,iter)];
            cdf(:,iter) = cdfCopulaStudent(u,rho_matrix,nu);
        end
    end
end
