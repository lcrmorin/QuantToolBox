function [Phi_CC,B_CC,Xi_T,B_T,results] = dice_temperature_matrix(Delta_t,scale,mtd)

    if nargin == 1
        scale = 365.25 * 24 * 3600;
    end

    phi1 = 0.2727;
    Phi_CC = [0.9120  0.0383       0;
              0.0880  0.9592  0.0003;
                   0  0.0025  0.9997];
    B_CC = [phi1; 0; 0];

    Xi_T_5 = [86.30  0.8624;
               2.50   97.50]/100;
    B_T_5  = [0.098; 0];
    results.Xi_T_5 = Xi_T_5;
    results.B_T_5 = B_T_5;

    Delta_5 = 5;
    Delta_1_seconds = scale;
    Delta_5_seconds = Delta_5 * Delta_1_seconds;
    Delta_t_seconds = Delta_t * Delta_1_seconds;

    xi1 = 0.098;
    xi2 = 3.8/2.9;
    xi3 = 0.088;
    xi4 = 0.025;
    c_AT = Delta_5_seconds / xi1;
    lambda = xi2;
    beta = xi3;
    c_LO = Delta_5_seconds * beta / xi4;

    xi1_prime = 1 - (lambda + beta) * Delta_t_seconds / c_AT;
    xi2_prime = beta * Delta_t_seconds / c_AT;
    xi3_prime = beta * Delta_t_seconds / c_LO;
    xi4_prime = 1 - beta * Delta_t_seconds / c_LO;
    Xi_T = [xi1_prime xi2_prime;
            xi3_prime xi4_prime];
    B_T = [Delta_t_seconds/c_AT; 0];

    results.xi1 = xi1;
    results.xi2 = xi2;
    results.xi3 = xi3;
    results.xi4 = xi4;

    results.c_AT = c_AT;
    results.c_LO = c_LO;
    results.lambda = lambda;
    results.beta = beta;

    results.xi1_prime = xi1_prime;
    results.xi2_prime = xi2_prime;
    results.xi3_prime = xi3_prime;
    results.xi4_prime = xi4_prime;

    if nargin == 3
        if mtd == 2
            Xi_T = mpower(Xi_T_5,1/5);
        end
    end

end
