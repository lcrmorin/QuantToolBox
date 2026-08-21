function PD = PD_Extended_Merton_Model(A0, D, r, mu_A, delta0, mu_delta, sigma_A, sigma_delta, rho, T)
% MERTON_PD  Probability of default in the extended Merton model (Blasberg-2024).
%
%   PD = MERTON_PD(A0, D, r, mu_A, delta0, mu_delta, sigma_A, sigma_delta, rho, T)
%
%   Computes the physical (P-measure) probability of default F(T):
%
%     F(T) = Phi( [ln(D) - ln(A0) - (mu_A - delta0 - 0.5*sigma_A^2)*T
%                  + 0.5*mu_delta*T^2] / (sigma_A' * sqrt(T)) )
%
%   Equivalently,  F(T) = Phi(-DD(T))  where DD(T) is the distance-to-default:
%
%     DD(T) = [ln(A0) - ln(D) + (mu_A - delta0 - 0.5*sigma_A^2)*T
%              - 0.5*mu_delta*T^2] / (sigma_A' * sqrt(T))
%
%   Inputs:
%     A0          - Current asset value
%     D           - Debt face value (default threshold)
%     r           - Risk-free interest rate (not used in PD, kept for
%                   consistent signature with merton_E0 / merton_B0)
%     mu_A        - Drift of the asset value under P
%     delta0      - Initial value of the growth adjustment factor delta(0)
%     mu_delta    - Drift of the growth adjustment factor
%     sigma_A     - Volatility of the asset value
%     sigma_delta - Volatility of the growth adjustment factor
%     rho         - Correlation: E[W1(t)*W2(t)] = rho*t
%     T           - Time horizon (debt maturity)
%
%   Output:
%     PD          - Probability of default in [0, 1]
%
%   Note: The variance of ln A(T) is the same under P and Q; only the mean
%   changes (mu_A replaces r). The rho*sigma_A*sigma_delta term does NOT
%   appear in the numerator of DD because it cancels: it enters the P-mean
%   of ln A(T) only through the variance correction, which is already
%   captured by sigma_A'.

% --- reparametrised effective volatility (same under P and Q) ---
sigma_A_prime = sqrt(sigma_A.^2 ...
                    - rho .* sigma_A .* sigma_delta .* T ...
                    + (1/3) .* sigma_delta.^2 .* T.^2);

% --- distance-to-default (under P) ---
DD = (log(A0 ./ D) + (mu_A - delta0 - 0.5 .* sigma_A^2) .* T - 0.5 .* mu_delta .* T.^2) ...
     ./ (sigma_A_prime .* sqrt(T));

% --- probability of default ---
PD = normcdf(-DD);

end
