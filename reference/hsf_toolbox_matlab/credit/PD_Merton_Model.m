function [PD_tau,A0,sigma_A,S_tau,DD_tau] = PD_Merton_Model(E0,sigma_E,D,mu_A,r,T,tau)

    n = max([rows(E0); rows(sigma_E); rows(D); rows(mu_A); rows(r); rows(T)]);
    e = ones(n,1);
    E0 = E0 .* e;
    sigma_E = sigma_E .* e;
    D = D .* e;
    mu_A = mu_A .* e;
    r = r .* e;
    T = T .* e;

    A0 = NaN(n,1);
    sigma_A = NaN(n,1);

    for i = 1:n
        sv = [E0(i); sigma_E(i)];
        extra_params = [E0(i); sigma_E(i); D(i); r(i); T(i)];
        obj = @(params) PD_Merton_Model_fn(params,extra_params);
        options = optimset('Display','off','TolFun',1e-10,'TolX',1e-10);
        [params,fmin,exitflag] = fminunc(obj,sv,options);
        params = abs(params);
        A0(i) = params(1);
        sigma_A(i) = params(2);
    end

    DD_tau = (log(A0 ./ D) + (mu_A - 0.5 .* sigma_A.^2) .* tau) ./ (sigma_A .* sqrt(tau));
    S_tau = normcdf(DD_tau);
    PD_tau = 1 -S_tau;
end

function obj = PD_Merton_Model_fn(params, extra_params)
    params = sqrt(params.^2);
    A0 = params(1);
    sigma_A = params(2);

    E0 = extra_params(1);
    sigma_E = extra_params(2);
    D = extra_params(3);
    r = extra_params(4);
    T = extra_params(5);

    d1 = (log(A0/D) + (r + 0.5*sigma_A^2)*T) / (sigma_A*sqrt(T));
    d2 = d1 - sigma_A*sqrt(T);

    obj1 = A0*normcdf(d1) - exp(-r*T)*D*normcdf(d2) - E0;
    obj2 = sigma_E*E0 - sigma_A*A0*normcdf(d1);

    obj = obj1 .^ 2 + obj2 .^2;
end
