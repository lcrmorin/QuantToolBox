function [E0, B0, k] = Merton_Jump_Climate_Model(A0,D,sigma_A,T,r,lambda,mu_Z,sigma_Z)

    k = exp(mu_Z + 0.5 .* sigma_Z.^2) - 1.0;
    E0 = 0;
    B0 = 0;
    n_max = max(500,ceil(lambda * T + 4 * sqrt(lambda * T)));
    p_n = exp(-lambda .* T);

    for n = 0:n_max
        b_n = r - lambda .* k + n .* log(1 + k) ./ T;
        sigma_n = sqrt(sigma_A.^2 + (n .* sigma_Z.^2) ./ T);
        d1_n = (log(A0 ./ D) + (b_n + 0.5*sigma_n.^2).*T) ./ (sigma_n.*sqrt(T));
        d2_n = d1_n - sigma_n.*sqrt(T);

        E_n = A0 .* exp((b_n-r).*T).*normcdf(d1_n) - D .* exp(-r.*T) .* normcdf(d2_n);
        B_n = A0 .* exp((b_n-r).*T).*normcdf(-d1_n) + D .* exp(-r.*T) .* normcdf(d2_n);

        E0 = E0 + p_n .* E_n;
        B0 = B0 + p_n .* B_n;
        p_n = p_n .* (lambda .* T) ./ (n+1);
        if lambda == 0
            break
        end
    end
end
