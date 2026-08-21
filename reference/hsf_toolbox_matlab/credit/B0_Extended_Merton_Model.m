function B0 = B0_Extended_Merton_Model(A0, D, r, mu_A, delta0, mu_delta, sigma_A, sigma_delta, rho, T)
% MERTON_B0  Bond (debt) value in the extended Merton model (Blasberg-2024).
%
%   B0 = MERTON_B0(A0, D, r, mu_A, delta0, mu_delta, sigma_A, sigma_delta, rho, T)
%
%   Computes the bond value at t = 0:
%
%     B0 = A0 * exp(-delta0_prime * T) * Phi(-d1) + D * exp(-r*T) * Phi(d2)
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
%     B0          - Bond value at t = 0
%
%   Balance-sheet check: E0 + B0 = A0 * exp(-delta0_prime * T)
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

% --- bond value ---
B0 = A0 .* exp(-delta0_prime .* T) .* normcdf(-d1) + D  .* exp(-r .* T) .* normcdf(d2);

end
