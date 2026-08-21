function [Call, Put, k] = Merton_Jump_Model(S0,K,sigma,T,b,r,lambda,mu_Z,sigma_Z)

    k = exp(mu_Z + 0.5 .* sigma_Z.^2) - 1.0;
    Call = 0;
    Put = 0;
    n_max = max(50,ceil(lambda * T + 4 * sqrt(lambda * T)));
    p_n = exp(-lambda .* T);

    for n = 0:n_max
        b_n = b - lambda .* k + n .* log(1 + k) ./ T;
        sigma_n = sqrt(sigma.^2 + (n .* sigma_Z.^2) ./ T);
        [Call_BS, Put_BS] = Black_Scholes_Model(S0,K,sigma_n,T,b_n,r);
        Call = Call + p_n .* Call_BS;
        Put = Put + p_n .* Put_BS;
        p_n = p_n .* (lambda .* T) ./ (n+1);
    end
end
