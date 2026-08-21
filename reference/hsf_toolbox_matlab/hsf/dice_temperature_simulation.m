function results = dice_temperature_simulation(t_0,t_end,Delta_t,Y_fn,mu_fn,parameters,numeric)

    nIters = (t_end - t_0)/Delta_t;

    CE_Land_0 = 3.3;
    delta_Land = 0.20;
    sigma_0 = 0.5491;
    g_sigma_0 = 0.01;
    delta_sigma = 0.001;

    CC_AT_0 = 830.4;
    CC_UP_0 = 1527;
    CC_LO_0 = 10010;

    CC_0 = [CC_AT_0; CC_UP_0; CC_LO_0];

    phi1 = 0.2727;
    Phi_CC = [0.9120  0.0383       0;
              0.0880  0.9592  0.0003;
                   0  0.0025  0.9997];
    B_CC = [phi1; 0; 0];

    eta = 3.8;
    CC_AT_1750 = 588;
    F_EX_0 = 0.25;
    F_EX_2100 = 0.70;
    Delta_F_EX = 5*(F_EX_2100 - F_EX_0)/90;
    F_RAD_0 = (eta/log(2))*log(CC_AT_0/CC_AT_1750) + F_EX_0;

    T_AT_0 = 0.8;
    T_LO_0 = 0.0068;

    T_0 = [T_AT_0; T_LO_0];

    xi1 = 0.098;
    xi2 = 3.8/2.9;
    xi3 = 0.088;
    xi4 = 0.025;
    c_AT = 5/xi1;
    lambda = xi2;
    beta = xi3;
    c_LO = 5 * beta / xi4;

    xi_prime1 = 1 - (lambda + beta) * Delta_t / c_AT;
    xi_prime2 = beta * Delta_t / c_AT;
    xi_prime3 = beta * Delta_t / c_LO;
    xi_prime4 = 1 - beta * Delta_t / c_LO;
    Xi_T = [xi_prime1 xi_prime2;
            xi_prime3 xi_prime4];
    B_T = [1/c_AT*Delta_t; 0];

    [Phi_CC,B_CC,Xi_T,B_T] = dice_temperature_matrix(Delta_t,1);

    if nargin == 7
        if numeric == 1
            Xi_T = [86.30  0.8624;
                     2.50   97.50]/100;
            B_T = [0.098; 0];
        end
    end

    if nargin > 5

    end

    Y_0 = Y_fn(t_0);
    mu_0 = mu_fn(t_0);

    CE_Industry_0 = (1-mu_0)*sigma_0*Y_0;
    CE_0 = CE_Industry_0 + CE_Land_0;

    t = t_0;
    sigma_t = sigma_0;
    CE_Land_t = CE_Land_0;
    CE_t = CE_0;
    g_sigma_t = g_sigma_0;
    CC_t = CC_0;
    F_EX_t = F_EX_0;
    F_RAD_t = F_RAD_0;
    T_t = T_0;

    results = zeros(nIters+1,10);
    results(1,:) = [t CE_t sigma_t CC_t' F_EX_t F_RAD_t T_t'];

    for iter = 2:nIters+1

        t = t + Delta_t;

        Y_t = Y_fn(t);
        mu_t = mu_fn(t);

        % Equation (8.4)
        CE_Industry_t = (1-mu_t)*sigma_t*Y_t;
        CE_Land_t = CE_Land_t * (1 - delta_Land);
        CE_t = CE_Industry_t + CE_Land_t;

        % Equation (8.5)
        g_sigma_t = 1/(1+delta_sigma) * g_sigma_t;
        sigma_t = (1 + g_sigma_t) * sigma_t;

        % Equation (8.6)
        CC_t = Phi_CC * CC_t + B_CC * CE_t;
        CC_AT_t = CC_t(1);

        % Equation (8.7)
        if t <= 2100
            F_EX_t = F_EX_t + Delta_F_EX;
        end
        F_RAD_t = (eta/log(2))*log(CC_AT_t/CC_AT_1750) + F_EX_t;

        % Equation (8.8)
        T_t = Xi_T * T_t + B_T * F_RAD_t;
        results(iter,:) = [t CE_t sigma_t CC_t' F_EX_t F_RAD_t T_t'];
    end

end
