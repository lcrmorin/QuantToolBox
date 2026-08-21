function E0 = E0_Extended_Merton_Model(A0, D, r, mu_A, delta0, mu_delta, sigma_A, sigma_delta, rho, T)
% MERTON_E0  Equity value in the extended Merton model (Blasberg-2024).
%
%   E0 = MERTON_E0(A0, D, r, mu_A, delta0, mu_delta, sigma_A, sigma_delta, rho, T)
%
%   Computes the equity value at t = 0:
%
%     E0 = A0 * exp(-delta0_prime * T) * Phi(d1) - D * exp(-r*T) * Phi(d2)
%
%   Inputs:
%     A0          - Current asset value
%     D           - Debt face value (default threshold)
%     r           - Risk-free interest rate
%     mu_A        - Drift of the asset value under P (not used here, kept
%                   for consistent signature with merton_PD)
%     delta0      - Initial value of the growth adjustment factor delta(0)
%     mu_delta    - Drift of the growth adjustment factor
%     sigma_A     - Volatility of the asset value
%     sigma_delta - Volatility of the growth adjustment factor
%     rho         - Correlation: E[W1(t)*W2(t)] = rho*t
%     T           - Time horizon (debt maturity)
%
%   Output:
%     E0          - Equity value at t = 0
%
%   Model:
%     dA(t) = (mu_A - delta(t)) * A(t) dt + sigma_A * A(t) dW1(t)
%     d(delta(t)) = mu_delta dt + sigma_delta dW2(t)
%
%   Reparametrised quantities (see text):
%     sigma_A'  = sqrt(sigma_A^2 - rho*sigma_A*sigma_delta*T
%                      + (1/3)*sigma_delta^2*T^2)
%     delta_0'  = delta0 + (1/2)*mu_delta*T
%                        + (1/2)*rho*sigma_A*sigma_delta*T
%                        - (1/6)*sigma_delta^2*T^2

% --- reparametrised volatility and drift adjustment ---
sigma_A_prime = sqrt(sigma_A.^2 ...
                    - rho .* sigma_A .* sigma_delta .* T ...
                    + (1/3) .* sigma_delta.^2 .* T.^2);
delta0_prime  = delta0 ...
              + 0.5 .* mu_delta .* T ...
              + 0.5 .* rho .* sigma_A .* sigma_delta .* T ...
              - (1/6) .* sigma_delta.^2 .* T.^2;

% --- d1 and d2 ---
d1 = (log(A0 ./ D) + r .* T - delta0_prime .* T) ./ (sigma_A_prime .* sqrt(T)) ...
     + 0.5 .* sigma_A_prime .* sqrt(T);
d2 = d1 - sigma_A_prime .* sqrt(T);

% --- equity value ---
E0 = A0 .* exp(-delta0_prime .* T) .* normcdf(d1) - D  .* exp(-r .* T) .* normcdf(d2);

end
