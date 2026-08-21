function [CB1,CB2] = carbon_budget_compound_reduction(t0,t,Delta_R,R_minus,CE_t0,g_Y)

if nargin == 5
    fun = @(s) CE_t0 .* (1-R_minus) .* ((1-Delta_R).^(s-t0));
    CB1 = integral(fun,t0,t);
    CB2 = ((1 - Delta_R) .^ (t-t0) - 1) ./ log(1 - Delta_R) .* (1 - R_minus) .* CE_t0;
else
    fun = @(s) CE_t0 .* (1-R_minus) .* ((1-Delta_R).^(s-t0)) .* ((1 + g_Y).^(s-t0));
    CB1 = integral(fun,t0,t);
    CB2 = ((1 + g_Y) .^ (t-t0) .* (1 - Delta_R) .^ (t-t0) - 1) ./ ...
        (log(1 + g_Y) + log(1 - Delta_R)) .* (1 - R_minus) .* CE_t0;
end

end
