function [CB1,CB2] = carbon_budget_linear_reduction(t0,t,Reduction,CE_t0)

fun = @(s) (1 - Reduction .* (s-t0)) .* CE_t0;
CB1 = integral(fun,t0,t);
CB2 = (t-t0) .* CE_t0 - (1/2) .* Reduction .* ((t-t0).^2) .* CE_t0;

end
