function [Loss,D_Loss,D2_Loss,results] = Reinders_Credit_Model(xi, A0, D, r, mu_A, sigma_A, T,omega_E, omega_D)

d1 = (log(A0 ./ D) + r .* T) ./ (sigma_A .* sqrt(T)) + 0.5 .* sigma_A .* sqrt(T);
d2 = d1 - sigma_A .* sqrt(T);
MV_E_t0 = A0 .* normcdf(d1)  - D  .* exp(-r .* T) .* normcdf(d2);
MV_D_t0 = A0 .* normcdf(-d1) + D  .* exp(-r .* T) .* normcdf(d2);

A_t = A0 .* (1 - xi);
d1 = (log(A_t ./ D) + r .* T) ./ (sigma_A .* sqrt(T)) + 0.5 .* sigma_A .* sqrt(T);
d2 = d1 - sigma_A .* sqrt(T);
MV_E_t = A_t .* normcdf(d1) - D  .* exp(-r .* T) .* normcdf(d2);
MV_D_t = A_t .* normcdf(-d1) + D  .* exp(-r .* T) .* normcdf(d2);

Loss = omega_E .* (MV_E_t0 - MV_E_t) + omega_D .* (MV_D_t0 - MV_D_t);
D_Loss = A0 .* (omega_E .* normcdf(d1) + omega_D .* normcdf(-d1));
D2_Loss = A0 .* (omega_D - omega_E) .* normpdf(d1) ./ (1 - xi) ./ (sigma_A .* sqrt(T));

results.MV_E_t0 = MV_E_t0;
results.MV_D_t0 = MV_D_t0;
results.MV_E_t = MV_E_t;
results.MV_D_t = MV_D_t;

end
